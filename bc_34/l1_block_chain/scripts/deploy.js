const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  const factory = await hre.ethers.getContractFactory("UAV_Registry", deployer);
  const contract = await factory.deploy();
  await contract.waitForDeployment();

  const address = await contract.getAddress();
  const deploymentDir = path.join(__dirname, "..", "deployments", "localhost");
  fs.mkdirSync(deploymentDir, { recursive: true });

  const outputPath = path.join(deploymentDir, "UAV_Registry.address.json");
  fs.writeFileSync(
    outputPath,
    JSON.stringify(
      {
        contractName: "UAV_Registry",
        address,
        chainId: (await hre.ethers.provider.getNetwork()).chainId.toString(),
      },
      null,
      2,
    )
  );

  console.log(`UAV_Registry deployed to ${address}`);
  console.log(`Deployment address written to ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
