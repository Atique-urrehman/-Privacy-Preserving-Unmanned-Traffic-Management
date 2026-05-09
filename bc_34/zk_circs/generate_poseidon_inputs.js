// Generates 8-field elements and computes an 8-leaf Merkle root using Poseidon
// Writes inputs.json compatible with batch_proof.circom
const fs = require('fs');
const path = require('path');

(async () => {
  try {
    const circomlib = require('circomlibjs');
    const poseidon = circomlib.poseidon;
    const F = poseidon.F;

    const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');

    function randField() {
      const buf = require('crypto').randomBytes(32);
      const n = BigInt('0x' + buf.toString('hex')) % fieldModulus;
      return n;
    }

    // If a seed file exists (seed_leaves.json) use those leaves (decimal strings).
    const seedPath = path.join(__dirname, 'seed_leaves.json');
    let leaves = [];
    if (fs.existsSync(seedPath)) {
      const seed = JSON.parse(fs.readFileSync(seedPath, 'utf8'));
      if (Array.isArray(seed.leaves) && seed.leaves.length >= 1) {
        leaves = seed.leaves.map(s => BigInt(s));
        console.log('✓ Using seeded leaves from seed_leaves.json');
      }
    }

    if (leaves.length === 0) {
      for (let i = 0; i < 8; i++) {
        leaves.push(randField());
      }
    }

    function poseidonHash(a, b) {
      // circomlibjs.poseidon returns a BigInt directly
      return BigInt(poseidon([a, b]));
    }

    // Build Merkle tree
    let level = leaves.slice();
    while (level.length > 1) {
      if (level.length % 2 === 1) level.push(level[level.length - 1]);
      const next = [];
      for (let i = 0; i < level.length; i += 2) {
        next.push(poseidonHash(level[i], level[i+1]));
      }
      level = next;
    }

    const root = level[0];

    const input = {
      leaves: leaves.map(x => x.toString()),
      new_root: root.toString()
    };

    fs.writeFileSync(path.join(__dirname, 'inputs.json'), JSON.stringify(input, null, 2));
    console.log('✓ Poseidon inputs generated and saved to inputs.json');
  } catch (e) {
    console.error('Failed to generate Poseidon inputs:', e);
    process.exit(1);
  }
})();
