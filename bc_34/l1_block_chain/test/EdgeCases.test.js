const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Edge Cases", function () {
  let verifier;
  let registry;
  let owner;
  let addr1;

  const validProof = {
    a: [
      15214717640235652803476763761039169046289203022831738008614935827959816769319n,
      5308696051485887305346050318848046387141370814155889388865649031055968847943n,
    ],
    b: [
      [
        10571949589435108814699897823196050399696405738065108803873814491883159815858n,
        18034865222996662996525900050357357211130635055253857485889607935502253262897n,
      ],
      [
        18309825051651652887093638381816948025049074609631051030813025883215555309652n,
        15969215308889319876046277234609313936769847289491003854076652945881551627968n,
      ],
    ],
    c: [
      9184815537885207525701213261862876820699485039066606897263051706897197505768n,
      2869996968309357524625066891076502882476662969905030699935621319606857905129n,
    ],
    publicInputs: [123456789012345678901234567890123456789012n],
  };

  beforeEach(async () => {
    [owner, addr1] = await ethers.getSigners();

    const Verifier = await ethers.getContractFactory("Groth16Verifier");
    verifier = await Verifier.deploy();
    await verifier.waitForDeployment();

    const Registry = await ethers.getContractFactory("UAV_Registry");
    registry = await Registry.deploy();
    await registry.waitForDeployment();
  });

  it("rejects a public signal outside the field", async () => {
    const fieldSize = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;

    expect(
      await verifier.verifyProof(validProof.a, validProof.b, validProof.c, [fieldSize])
    ).to.equal(false);
  });

  it("rejects a zero proof", async () => {
    const zeroProof = {
      a: [0n, 0n],
      b: [[0n, 0n], [0n, 0n]],
      c: [0n, 0n],
    };

    expect(
      await verifier.verifyProof(zeroProof.a, zeroProof.b, zeroProof.c, validProof.publicInputs)
    ).to.equal(false);
  });

  it("updates the batch root when submitBatch is called", async () => {
    const newRoot = ethers.keccak256(ethers.toUtf8Bytes("batch-1"));

    await expect(registry.submitBatch("0x1234", newRoot)).to.emit(registry, "StateUpdated");

    expect(await registry.currentStateRoot()).to.equal(newRoot);
  });

  it("allows duplicate roots and keeps state stable", async () => {
    const root = ethers.keccak256(ethers.toUtf8Bytes("duplicate-root"));

    await registry.connect(owner).submitBatch("0x1234", root);
    await registry.connect(addr1).submitBatch("0x1234", root);

    expect(await registry.currentStateRoot()).to.equal(root);
  });

  it("supports sequential submissions from different signers", async () => {
    const firstRoot = ethers.keccak256(ethers.toUtf8Bytes("first-root"));
    const secondRoot = ethers.keccak256(ethers.toUtf8Bytes("second-root"));

    await registry.connect(owner).submitBatch("0x1234", firstRoot);
    await registry.connect(addr1).submitBatch("0x1234", secondRoot);

    expect(await registry.currentStateRoot()).to.equal(secondRoot);
  });
});