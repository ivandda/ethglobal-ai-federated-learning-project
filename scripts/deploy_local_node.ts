import { network } from "hardhat";

const { ethers, networkName } = await network.connect();

console.log(`Deploying FederatedLearningLog to ${networkName}...`);

// Use two local accounts as owner and server
const [owner, server] = await ethers.getSigners();

console.log("Owner address:", owner.address);
console.log("Server address (FL server):", server.address);

// Deploy contract to the dev node
const fl = await ethers.deployContract("FederatedLearningLog", [server.address]);

console.log("Waiting for deployment tx to confirm...");
await fl.waitForDeployment();

const flAddress = await fl.getAddress();
console.log("FederatedLearningLog deployed at:", flAddress);

const storedServer = await fl.server();
console.log("Stored server in contract:", storedServer);

console.log("Deployment to dev network successful!");
