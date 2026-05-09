const { buildModule } = require("@nomicfoundation/hardhat-ignition/modules");

module.exports = buildModule("DeployUAVRegistry", (m) => {
  const registry = m.contract("UAV_Registry");
  return { registry };
});
