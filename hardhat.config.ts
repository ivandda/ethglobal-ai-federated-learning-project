import { defineConfig } from "hardhat/config";
import hardhatEthers from "@nomicfoundation/hardhat-ethers";

export default defineConfig({
  solidity: "0.8.28",
  plugins: [hardhatEthers],
  // No networks needed yet: we will use the built-in "default" network,
  // which is an edr-simulated in-memory chain.
});
