const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");
const { ethers } = require("ethers");
const { createZGComputeNetworkBroker } = require("@0glabs/0g-serving-broker");

// Load .env.0g from parent directory
dotenv.config({ path: path.join(__dirname, "..", ".env.0g") });

async function main() {
  const question = process.env.QUESTION || process.argv.slice(2).join(" ");
  if (!question) {
    console.error("Usage: QUESTION='your question' pnpm ask-run");
    process.exit(1);
  }

  const rpcUrl = process.env.RPC_ENDPOINT || "https://evmrpc-testnet.0g.ai";
  const privateKey = process.env.ZG_PRIVATE_KEY;

  if (!privateKey) {
    console.error("Missing ZG_PRIVATE_KEY in environment");
    console.error(`Tried to load from: ${path.join(__dirname, "..", ".env.0g")}`);
    process.exit(1);
  }

  // 1) Load summary data - try multiple methods automatically
  let summaryJson;
  const storageRoot = process.env.STORAGE_ROOT || process.env.SUMMARY_DATA_ROOT;
  const storageFilename = process.env.STORAGE_FILENAME || process.env.SUMMARY_DATA_FILENAME || "run_summary_data.json";
  const contractAddress = process.env.CONTRACT_ADDRESS;
  const contractRpcUrl = process.env.RPC_URL || rpcUrl;
  
  // Method 1: Try to download directly from storage if root hash is provided
  if (storageRoot) {
    try {
      const indexer = process.env.ZEROG_INDEXER || "https://indexer-storage-testnet-turbo.0g.ai/";
      const indexerUrl = indexer.endsWith("/") ? indexer.slice(0, -1) : indexer;
      const downloadUrl = `${indexerUrl}/file?root=${storageRoot}&name=${storageFilename}`;
      
      console.log(`Downloading summary data directly from 0G storage...`);
      console.log(`Root: ${storageRoot}`);
      console.log(`Filename: ${storageFilename}`);
      
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Download failed. Status: ${response.status}`);
        console.error(`Response preview: ${errorText.substring(0, 200)}`);
        throw new Error(`Failed to download from 0G: ${response.status} ${response.statusText}`);
      }
      
      summaryJson = await response.text();
      console.log(`Successfully downloaded ${summaryJson.length} bytes from 0G storage`);
      
      // Verify it's valid JSON
      try {
        JSON.parse(summaryJson);
      } catch (parseErr) {
        console.error(`Downloaded content is not valid JSON. First 200 chars: ${summaryJson.substring(0, 200)}`);
        console.error(`Content-Type: ${response.headers.get('content-type')}`);
        throw new Error(`Downloaded file is not valid JSON: ${parseErr.message}`);
      }
    } catch (err) {
      console.error(`Failed to download from 0G storage: ${err.message}`);
      console.error(`Please check STORAGE_ROOT and STORAGE_FILENAME environment variables`);
      process.exit(1);
    }
  } else if (contractAddress) {
    // Method 2: Query smart contract to get latest round's storage root
    console.log("Querying smart contract for latest round...");
    try {
      const provider = new ethers.JsonRpcProvider(contractRpcUrl);
      
      // Load contract ABI
      const artifactPath = path.join(__dirname, "..", "artifacts", "contracts", "FederatedLearningLog.sol", "FederatedLearningLog.json");
      if (!fs.existsSync(artifactPath)) {
        throw new Error(`Contract artifact not found at ${artifactPath}`);
      }
      const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf-8"));
      const contract = new ethers.Contract(contractAddress, artifact.abi, provider);
      
      // Get latest round ID
      const latestRoundId = await contract.latestRoundId();
      console.log(`Latest round ID: ${latestRoundId}`);
      
      if (latestRoundId === 0n) {
        throw new Error("No rounds recorded yet in the contract");
      }
      
      // Get round info
      const roundInfo = await contract.rounds(latestRoundId);
      const artifactCid = roundInfo.artifactCid;
      const summaryCid = roundInfo.summaryCid;
      console.log(`Found artifact CID from round ${latestRoundId}: ${artifactCid}`);
      console.log(`Found summary CID from round ${latestRoundId}: ${summaryCid || '(not set)'}`);
      
      // Use summaryCid if available, otherwise fall back to artifactCid (old behavior)
      const storageRootToUse = summaryCid && summaryCid.length > 0 ? summaryCid : artifactCid;
      
      if (!summaryCid || summaryCid.length === 0) {
        console.warn("Warning: summaryCid not set in contract. Using artifactCid (model file) instead.");
        console.warn("This round was recorded before the contract was updated to store summaryCid.");
      }
      
      // Try to download summary data using the summary CID (or artifact CID as fallback)
      const indexer = process.env.ZEROG_INDEXER || "https://indexer-storage-testnet-turbo.0g.ai/";
      const indexerUrl = indexer.endsWith("/") ? indexer.slice(0, -1) : indexer;
      const downloadUrl = `${indexerUrl}/file?root=${storageRootToUse}&name=${storageFilename}`;
      
      console.log(`Attempting to download summary from contract's ${summaryCid ? 'summary' : 'artifact'} CID...`);
      console.log(`URL: ${downloadUrl}`);
      
      const response = await fetch(downloadUrl);
      if (response.ok) {
        const contentType = response.headers.get('content-type') || '';
        summaryJson = await response.text();
        console.log(`Successfully downloaded ${summaryJson.length} bytes from 0G storage using contract data`);
        console.log(`Content-Type: ${contentType}`);
        
        // Verify it's valid JSON
        try {
          const parsed = JSON.parse(summaryJson);
          console.log("Downloaded content is valid JSON");
          
          // Check if this is actually the summary data (has expected fields)
          if (!parsed.rounds && !parsed.dataset) {
            console.warn("Downloaded JSON doesn't look like summary data. The artifact CID might point to the model file, not the summary.");
            throw new Error("Downloaded file doesn't contain summary data. Contract's artifactCid points to model file, not summary.");
          }
        } catch (parseErr) {
          // Check if it looks like a binary file (model file) or HTML error
          const firstChars = summaryJson.substring(0, 100);
          const isLikelyBinary = /[\x00-\x08\x0E-\x1F]/.test(firstChars) || 
                                 summaryJson.startsWith('PK') || // ZIP file
                                 summaryJson.startsWith('\x80'); // PyTorch pickle
          const isLikelyHTML = summaryJson.trim().startsWith('<');
          
          if (isLikelyBinary) {
            console.error(`Downloaded content appears to be a binary file (likely the model .pt file)`);
            console.error(`The contract's artifactCid (${artifactCid}) points to the model file, not the summary.`);
            throw new Error(`Contract's artifactCid points to model file (final_model.pt), not summary. Falling back to local file method.`);
          } else if (isLikelyHTML) {
            console.error(`Downloaded content appears to be HTML (error page). First 500 chars:`);
            console.error(summaryJson.substring(0, 500));
            throw new Error(`Received HTML error page instead of JSON. Summary file not found at artifact CID.`);
          } else {
            console.error(`Downloaded content is not valid JSON. First 500 chars:`);
            console.error(summaryJson.substring(0, 500));
            console.error(`Parse error: ${parseErr.message}`);
            throw new Error(`Downloaded file is not valid JSON: ${parseErr.message}`);
          }
        }
      } else {
        // If summary file not found, try to get it from local file or fallback
        const errorText = await response.text();
        console.warn(`Summary file not found at artifact CID. Status: ${response.status}`);
        console.warn(`Response preview: ${errorText.substring(0, 500)}`);
        throw new Error(`Summary file not found: ${response.status} ${response.statusText}. Note: Contract's artifactCid points to the model file, not the summary.`);
      }
    } catch (err) {
      console.warn(`Failed to get data from contract: ${err.message}`);
      console.warn(`Falling back to local file...`);
      // Continue to fallback method
    }
  }
  
  // Method 3: Fallback to local file
  if (!summaryJson) {
    const summaryPath = path.join(__dirname, "..", "fl-engine", "run_summary.json");
    if (!fs.existsSync(summaryPath)) {
      console.error("Could not load summary data from any source.");
      console.error("Tried:");
      console.error("  1. STORAGE_ROOT environment variable");
      if (contractAddress) {
        console.error("  2. Smart contract query (CONTRACT_ADDRESS set)");
      }
      console.error("  3. Local run_summary.json file");
      console.error("\nOptions:");
      console.error("  - Set STORAGE_ROOT environment variable");
      console.error("  - Set CONTRACT_ADDRESS to query on-chain");
      console.error("  - Run `flwr run .` first to generate run_summary.json");
      process.exit(1);
    }
    
    const summaryMetadata = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    
    // Try to download from storage using metadata
    if (summaryMetadata.summary_data_root && summaryMetadata.summary_data_filename) {
      try {
        const indexer = summaryMetadata.storage?.zero_g_indexer || 
                       process.env.ZEROG_INDEXER || 
                       "https://indexer-storage-testnet-turbo.0g.ai/";
        const indexerUrl = indexer.endsWith("/") ? indexer.slice(0, -1) : indexer;
        const downloadUrl = `${indexerUrl}/file?root=${summaryMetadata.summary_data_root}&name=${summaryMetadata.summary_data_filename}`;
        
        console.log(`Downloading summary data from 0G storage...`);
        console.log(`Root: ${summaryMetadata.summary_data_root}`);
        console.log(`Filename: ${summaryMetadata.summary_data_filename}`);
        
        const response = await fetch(downloadUrl);
        if (!response.ok) {
          const errorText = await response.text();
          console.error(`Download failed. Status: ${response.status}`);
          console.error(`Response preview: ${errorText.substring(0, 200)}`);
          throw new Error(`Failed to download from 0G: ${response.status} ${response.statusText}`);
        }
        
        summaryJson = await response.text();
        console.log(`Successfully downloaded ${summaryJson.length} bytes from 0G storage`);
        
        // Verify it's valid JSON
        try {
          JSON.parse(summaryJson);
        } catch (parseErr) {
          console.error(`Downloaded content is not valid JSON. First 200 chars: ${summaryJson.substring(0, 200)}`);
          console.error(`Content-Type: ${response.headers.get('content-type')}`);
          throw new Error(`Downloaded file is not valid JSON: ${parseErr.message}`);
        }
      } catch (err) {
        console.warn(`Failed to download from 0G storage: ${err.message}`);
        console.warn(`Falling back to local file...`);
        summaryJson = JSON.stringify(summaryMetadata, null, 2);
      }
    } else {
      // No 0G storage data, use local file
      console.log("No 0G storage data found, using local file");
      summaryJson = JSON.stringify(summaryMetadata, null, 2);
    }
  }

  // 3) Create broker
  console.log("Creating broker...");
  const provider = new ethers.JsonRpcProvider(rpcUrl);
  const wallet = new ethers.Wallet(privateKey, provider);

  const broker = await createZGComputeNetworkBroker(wallet);
  console.log("Broker created successfully");

  // 4) Pick a provider/model
  const providerAddress = process.env.ZG_PROVIDER_ADDRESS || "0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3";

  console.log(`Using provider address: ${providerAddress}`);
  console.log("Acknowledging provider...");
  await broker.inference.acknowledgeProviderSigner(providerAddress);

  // 5) Get service metadata
  console.log("Getting service metadata...");
  const { endpoint, model } = await broker.inference.getServiceMetadata(
    providerAddress
  );
  console.log(`Using endpoint: ${endpoint}, model: ${model}`);

  // 6) Filter summary to only essential data for LLM (reduce token count)
  let filteredSummary;
  try {
    // Check if summaryJson is already a string or object
    let fullSummary;
    if (typeof summaryJson === 'string') {
      // Try to parse JSON
      try {
        fullSummary = JSON.parse(summaryJson);
      } catch (parseErr) {
        // If parsing fails, check what we got
        const preview = summaryJson.substring(0, 200);
        console.error(`Failed to parse summary as JSON. Content preview: ${preview}`);
        console.error(`Parse error: ${parseErr.message}`);
        throw new Error(`Summary data is not valid JSON. This might be the model file or an error page. Error: ${parseErr.message}`);
      }
    } else {
      fullSummary = summaryJson;
    }
    
    // Extract only essential information
    filteredSummary = {
      num_server_rounds: fullSummary.num_server_rounds,
      dataset: {
        name: fullSummary.dataset?.name,
        num_classes: fullSummary.dataset?.num_classes,
        type: fullSummary.dataset?.type,
      },
      model: {
        architecture: fullSummary.model?.architecture,
        description: fullSummary.model?.description,
      },
      training_config: {
        learning_rate: fullSummary.training_config?.learning_rate,
        num_server_rounds: fullSummary.training_config?.num_server_rounds,
        batch_size: fullSummary.training_config?.batch_size,
      },
      global_metrics: fullSummary.global_metrics,
      rounds: fullSummary.rounds?.map(round => ({
        round: round.round,
        metrics: {
          train_loss: round.metrics?.train_loss,
          eval_loss: round.metrics?.eval_loss,
          eval_acc: round.metrics?.eval_acc,
        }
      })) || [],
    };
    
    console.log(`Filtered summary: ${JSON.stringify(filteredSummary).length} bytes (from ${summaryJson.length} bytes)`);
  } catch (err) {
    console.warn(`Failed to filter summary, using full data: ${err.message}`);
    filteredSummary = summaryJson; // Fallback to full data
  }

  // 7) Build the prompt from filtered summary + question
  const systemPrompt = `
You are an assistant helping interpret a federated learning run.
You will be given a JSON summary of the last run and a user question.

The summary includes:
- Dataset information (name, type, classes)
- Model architecture details
- Training configuration (learning rate, epochs, etc.)
- Round-by-round metrics showing:
  * Training loss per round
  * Evaluation accuracy per round
  * Accuracy improvement between rounds
- Overall model improvement metrics

When analyzing the data:
1. Explain how the model improved over rounds (accuracy trends)
2. Highlight key metrics and improvements
3. Use the round-by-round data to show progression

Use ONLY the provided JSON and general ML knowledge. If something
isn't in the data, say so explicitly.
`;

  const messages = [
    { role: "system", content: systemPrompt },
    {
      role: "user",
      content:
        "Here is the JSON summary of the last run:\n\n" +
        JSON.stringify(filteredSummary, null, 2) +
        "\n\nQuestion: " +
        question,
    },
  ];

  // 8) Get request headers for this single request
  console.log("Getting request headers...");
  const headers = await broker.inference.getRequestHeaders(
    providerAddress,
    JSON.stringify(messages)
  );

  // 9) Call the 0G LLM endpoint
  console.log("Calling LLM endpoint...");
  const response = await fetch(`${endpoint}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify({
      messages,
      model,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("LLM request failed:", response.status, errorText);
    process.exit(1);
  }

  const data = await response.json();
  const answer = data.choices?.[0]?.message?.content ?? "(no answer)";
  const chatID = data.id;

  console.log("\n=== LLM Answer ===\n");
  console.log(answer);
  console.log("\n==================\n");

  // 10) Optional: verify response
  try {
    const isValid = await broker.inference.processResponse(
      providerAddress,
      answer,
      chatID
    );
    console.log(`Verification: ${isValid ? "VALID" : "NOT VERIFIED"}`);
  } catch (err) {
    console.log("Verification failed/unsupported:", err.message);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});