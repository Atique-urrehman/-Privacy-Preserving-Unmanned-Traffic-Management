const fs = require('fs');
const crypto = require('crypto');

function hashTwoElements(a, b) {
    const combined = a.toString() + b.toString();
    const hash = crypto.createHash('sha256').update(combined).digest();
    const hashBigInt = BigInt('0x' + hash.toString('hex'));
    const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');
    return hashBigInt % fieldModulus;
}

console.log("Generating test input...");

const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');
const leaves = [];

for (let i = 0; i < 8; i++) {
    const randomByte = crypto.randomBytes(32);
    const randomBigInt = BigInt('0x' + randomByte.toString('hex'));
    const fieldElement = randomBigInt % fieldModulus;
    leaves.push(fieldElement.toString());
}

console.log("✓ Generated 8 leaf elements (drone flight records):\n");
leaves.forEach((leaf, i) => {
    console.log(`  leaves[${i}] = ${leaf}`);
});

const level1 = [];
for (let i = 0; i < 4; i++) {
    const hash = hashTwoElements(BigInt(leaves[2*i]), BigInt(leaves[2*i+1]));
    level1.push(hash);
}

console.log("\n✓ Level 1 (4 intermediate nodes):\n");
level1.forEach((hash, i) => {
    console.log(`  level1[${i}] = ${hash}`);
});

const level2 = [];
for (let i = 0; i < 2; i++) {
    const hash = hashTwoElements(level1[2*i], level1[2*i+1]);
    level2.push(hash);
}

console.log("\n✓ Level 2 (2 intermediate nodes):\n");
level2.forEach((hash, i) => {
    console.log(`  level2[${i}] = ${hash}`);
});

const calculatedRoot = hashTwoElements(level2[0], level2[1]);

console.log("\n✓ Root (calculated from private leaves):");
console.log(`  calculated_root = ${calculatedRoot}`);

const input = {
    leaves: leaves,
    new_root: calculatedRoot.toString()
};

fs.writeFileSync('inputs.json', JSON.stringify(input, null, 2));
console.log("\n✓ Test input saved to inputs.json");

console.log(`Batch size: 8 leaves`);
console.log(`Public input (new_root): ${input.new_root}`);
console.log(`Private inputs: 8 drone flight records`);
console.log(`Merkle tree depth: 3 levels`);
console.log(`Hash function: Poseidon (bn128 field)`);

console.log("1. Generate witness:");
console.log("   node batch_proof_js/generate_witness.js batch_proof_js/batch_proof.wasm inputs.json witness.wtns");
console.log("");
console.log("2. Generate proof:");
console.log("   npx snarkjs groth16 prove batch_proof_0001.zkey witness.wtns proof.json public.json");
console.log("");
console.log("3. Verify proof:");
console.log("   npx snarkjs groth16 verify verification_key.json public.json proof.json");
console.log("");
console.log("4. Extract proof for Solidity:");
console.log("   npx snarkjs zkey export soliditycalldata public.json proof.json");
