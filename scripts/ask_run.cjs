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

  // 1) Load the run summary
  const summaryPath = path.join(__dirname, "..", "fl-engine", "run_summary.json");
  if (!fs.existsSync(summaryPath)) {
    console.error("run_summary.json not found. Run `flwr run .` first.");
    console.error(`Looking for: ${summaryPath}`);
    process.exit(1);
  }
  const summaryJson = fs.readFileSync(summaryPath, "utf-8");

  // 2) Create broker
  console.log("Creating broker...");
  const provider = new ethers.JsonRpcProvider(rpcUrl);
  const wallet = new ethers.Wallet(privateKey, provider);

  const broker = await createZGComputeNetworkBroker(wallet);
  console.log("Broker created successfully");

  // 3) Pick a provider/model
  const providerAddress = "0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3";

  console.log("Acknowledging provider...");
  await broker.inference.acknowledgeProviderSigner(providerAddress);

  // 4) Get service metadata
  console.log("Getting service metadata...");
  const { endpoint, model } = await broker.inference.getServiceMetadata(
    providerAddress
  );
  console.log(`Using endpoint: ${endpoint}, model: ${model}`);

  // 5) Build the prompt from summary + question
  const systemPrompt = `
You are an assistant helping interpret a federated learning run.
You will be given a JSON summary of the last run and a user question.
Use ONLY the provided JSON and general ML knowledge. If something
isn't in the data, say so explicitly.
`;

  const messages = [
    { role: "system", content: systemPrompt },
    {
      role: "user",
      content:
        "Here is the JSON summary of the last run:\n\n" +
        summaryJson +
        "\n\nQuestion: " +
        question,
    },
  ];

  // 6) Get request headers for this single request
  console.log("Getting request headers...");
  const headers = await broker.inference.getRequestHeaders(
    providerAddress,
    JSON.stringify(messages)
  );

  // 7) Call the 0G LLM endpoint
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

  // 8) Optional: verify response
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