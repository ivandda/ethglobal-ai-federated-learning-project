import { defineConfig } from "hardhat/config";
import hardhatEthers from "@nomicfoundation/hardhat-ethers";

export default defineConfig({
  solidity: "0.8.28",
  plugins: [hardhatEthers],
  networks: {
    // Default in-process network (no flag): "default"
    // Extra HTTP network pointing to a local Hardhat node:
    dev: {
      type: "http",
      url: "http://127.0.0.1:8545",
    },
  },
});
