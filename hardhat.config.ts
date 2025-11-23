import { defineConfig } from "hardhat/config";
import hardhatEthers from "@nomicfoundation/hardhat-ethers";
// import dotenv from "dotenv";
import * as dotenv from "dotenv";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";

// Get __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env.0g file (or .env as fallback)
dotenv.config({ path: resolve(__dirname, ".env.0g") });
dotenv.config({ path: resolve(__dirname, ".env") }); // Fallback to .env

export default defineConfig({
  solidity: {
    version: "0.8.28",
    settings: {
      evmVersion: "cancun",
      optimizer: {
        enabled: true,
        runs: 200,
      },
      viaIR: true, // Required to fix "stack too deep" error
    },
  },
  plugins: [hardhatEthers],
  networks: {
    // Default in-process network (no flag): "default"
    // Extra HTTP network pointing to a local Hardhat node:
    dev: {
      type: "http",
      url: "http://127.0.0.1:8545",
    },
    "0g-testnet": {
      type: "http",
      url: "https://evmrpc-testnet.0g.ai",
      accounts: process.env.PRIVATE_KEY_0G ? [process.env.PRIVATE_KEY_0G] : [],
    },
  },
});
