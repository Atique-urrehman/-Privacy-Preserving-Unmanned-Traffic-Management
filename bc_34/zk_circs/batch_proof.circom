pragma circom 2.0.0;

include "circomlib/circuits/poseidon.circom";

template BatchProof() {
    signal input leaves[8];
    
    signal input new_root;
    
    signal level1_out[4];
    
    component hasher_l1[4];
    
    for (var i = 0; i < 4; i++) {
        hasher_l1[i] = Poseidon(2);
        hasher_l1[i].inputs[0] <== leaves[2*i];
        hasher_l1[i].inputs[1] <== leaves[2*i + 1];
        level1_out[i] <== hasher_l1[i].out;
    }
    
    signal level2_out[2];
    
    component hasher_l2[2];
    
    for (var i = 0; i < 2; i++) {
        hasher_l2[i] = Poseidon(2);
        hasher_l2[i].inputs[0] <== level1_out[2*i];      // First node of pair
        hasher_l2[i].inputs[1] <== level1_out[2*i + 1];  // Second node of pair
        // Store the hash result
        level2_out[i] <== hasher_l2[i].out;
    }
    
    signal calculated_root;
    
    component hasher_root = Poseidon(2);
    hasher_root.inputs[0] <== level2_out[0];  // Left node at level 2
    hasher_root.inputs[1] <== level2_out[1];  // Right node at level 2
    calculated_root <== hasher_root.out;
    
    calculated_root === new_root;
}

component main {public [new_root]} = BatchProof();
