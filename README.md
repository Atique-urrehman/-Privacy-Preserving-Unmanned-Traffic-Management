# Privacy-Preserving-Unmanned-Traffic-Management

```
python3 -m venv .env
pip install web3 matplotlib 
pip install cryptography
pip install pycryptodome
```

## Initial Simulations

```
python phase2.py
python sim.py
```

## Complete Prototype Simulation

```
cd bc_34/zk_circs/
make all
mv Verifier.sol ../l1_block_chain/
cd ../l1_block_chain/
npm install
npx hardhat node # in one terminal
npx hardhat run scripts/deploy.js --network localhost # in another terminal
cd ../l2_edge/
python drone_fleet.py

```
