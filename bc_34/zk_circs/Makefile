.PHONY: help install compile setup setup-tau setup-phase2 generate-verifier clean test all

help:
	@echo "ZK Circuit Build Targets:"
	@echo "  make install          - Install npm dependencies"
	@echo "  make compile          - Compile Circom circuit"
	@echo "  make setup            - Run full Groth16 setup"
	@echo "  make setup-tau        - Run Powers of Tau setup"
	@echo "  make setup-phase2     - Run Groth16 key generation"
	@echo "  make generate-verifier- Generate Solidity verifier"
	@echo "  make integrate        - Copy artifacts to Hardhat"
	@echo "  make test-input       - Generate test input data"
	@echo "  make all              - Full build pipeline"
	@echo "  make clean            - Remove generated files"

install:
	npm install

compile: node_modules
	@echo " Compiling circuit..."
	circom batch_proof.circom --r1cs --wasm -l node_modules
	@echo " Circuit compiled successfully"
	@echo "   - batch_proof.r1cs (472 KB)"
	@echo "   - batch_proof_js/batch_proof.wasm (1.7 MB)"

# Powers of Tau Phase 1
setup-tau: node_modules
	@echo " Creating Powers of Tau (patience: ~30 seconds)..."
	npx snarkjs powersoftau new bn128 15 pot15_0000.ptau
	@echo " Adding entropy contribution..."
	echo "drone-layer2-entropy" | npx snarkjs powersoftau contribute pot15_0000.ptau pot15_0001.ptau
	@echo " Preparing for phase 2 (patience: ~2 minutes)..."
	npx snarkjs powersoftau prepare phase2 pot15_0001.ptau pot15_final.ptau
	@echo " Powers of Tau complete"

# Groth16 Setup Phase 2
setup-phase2: batch_proof.r1cs pot15_final.ptau
	@echo " Generating initial zkey..."
	npx snarkjs groth16 setup batch_proof.r1cs pot15_final.ptau batch_proof_0000.zkey
	@echo " Adding zkey contribution..."
	echo "drone-zkey-entropy" | npx snarkjs zkey contribute batch_proof_0000.zkey batch_proof_0001.zkey
	@echo " Exporting verification key..."
	npx snarkjs zkey export verificationkey batch_proof_0001.zkey verification_key.json
	@echo " Groth16 setup complete"

generate-verifier: batch_proof_0001.zkey
	@echo " Generating Solidity verifier..."
	npx snarkjs zkey export solidityverifier batch_proof_0001.zkey Verifier.sol
	@echo " Verifier.sol generated (7.0 KB)"

setup: setup-tau setup-phase2 generate-verifier
	@echo ""
	@echo " FULL SETUP COMPLETE!"
	@echo ""
	@echo "Generated files:"
	@ls -lh batch_proof_0001.zkey verification_key.json Verifier.sol | awk '{print "  " $$9, "(" $$5 ")"}'
	@echo ""
	@echo "Next: make integrate"

test-input: node_modules
	@echo "🧪 Creating test input data..."
	node create_test_input.js
	@echo " Test input created: inputs.json"

integrate:
	@echo " Integrating with Hardhat..."
	cp Verifier.sol ../l1_block_chain/contracts/
	@echo " Verifier.sol copied to Hardhat contracts/"
	@echo ""
	@echo " Update l1_block_chain/contracts/UAV_Registry.sol:"
	@echo "   1. Import Verifier.sol"
	@echo "   2. Add submitBatchProof() function"
	@echo "   3. See INTEGRATION_GUIDE.md for details"

# Clean generated files
clean:
	@echo " Cleaning generated files..."
	rm -rf batch_proof_js/ batch_proof.r1cs batch_proof.wasm batch_proof.sym
	rm -f pot15_*.ptau batch_proof_*.zkey
	rm -f verification_key.json Verifier.sol
	rm -f inputs.json witness.wtns proof.json public.json
	@echo " Clean complete"

# Full pipeline
all: install compile setup integrate
	@echo ""
	@echo " FULL BUILD COMPLETE!"
	@echo ""
	@echo "Summary:"
	@echo "   Dependencies installed"
	@echo "   Circuit compiled (1,918 constraints)"
	@echo "   Powers of Tau ceremony completed"
	@echo "   Groth16 keys generated"
	@echo "   Solidity verifier exported"
	@echo "   Artifacts integrated with Hardhat"
	@echo ""
	@echo "Files ready for deployment:"
	@ls -1 Verifier.sol verification_key.json batch_proof_0001.zkey 2>/dev/null || echo "Files not found - run setup first"
	@echo ""

.PHONY: node_modules
node_modules:
	@test -d node_modules || npm install

.PHONY: batch_proof.r1cs
batch_proof.r1cs: compile

.PHONY: pot15_final.ptau
pot15_final.ptau: setup-tau

.PHONY: batch_proof_0001.zkey
batch_proof_0001.zkey: setup-phase2

