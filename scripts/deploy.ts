import { network } from "hardhat";

const { ethers, networkName } = await network.connect();

async function main() {
  const signers = await ethers.getSigners();

  if (signers.length === 0) {
    throw new Error("No signers available. Check your 0g-testnet accounts config.");
  }

  const owner = signers[0];
  // If you want a separate server address, allow env override; otherwise reuse owner
  const serverAddress =
    process.env.FL_SERVER_ADDRESS && process.env.FL_SERVER_ADDRESS !== ""
      ? process.env.FL_SERVER_ADDRESS
      : owner.address;

  console.log(`Deploying FederatedLearningLog to ${networkName}...`);
  console.log("Owner address:", owner.address);
  console.log("Server address (FL server):", serverAddress);

  const FederatedLearningLog = await ethers.getContractFactory(
    "FederatedLearningLog",
    owner,
  );

  const contract = await FederatedLearningLog.deploy(serverAddress);

  console.log("Waiting for the deployment tx to confirm...");
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  console.log("FederatedLearningLog deployed at:", address);

  const storedServer = await contract.server();
  console.log("Stored server in contract:", storedServer);

  console.log("Deployment to 0g-testnet successful!");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
