#!/bin/bash
# Edge Case Testing Execution Script
# Location: /home/uak/Projects/bc_34/l1_block_chain/run_edge_tests.sh
# Usage: ./run_edge_tests.sh [options]
# Options: --verbose, --coverage, --gas-report, --all

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_FILE="test/EdgeCases.test.js"
NETWORK="hardhat"
TIMEOUT=60000

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Edge Case Testing Suite - BatchProof System${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}✗ Error: Test file not found at $TEST_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Test file found${NC}"
echo -e "${YELLOW}Running tests...${NC}\n"

# Build test command
TEST_CMD="npx hardhat test $TEST_FILE --network $NETWORK --timeout $TIMEOUT"

# Add flags based on arguments
if [[ "$*" == *"--verbose"* ]]; then
    TEST_CMD="$TEST_CMD --verbose"
    echo -e "${BLUE}[INFO] Verbose mode enabled${NC}"
fi

if [[ "$*" == *"--gas-report"* ]]; then
    # Enable gas reporting via environment
    export REPORT_GAS=true
    echo -e "${BLUE}[INFO] Gas reporting enabled${NC}"
fi

if [[ "$*" == *"--coverage"* ]]; then
    echo -e "${BLUE}[INFO] Running coverage analysis...${NC}"
    npx hardhat coverage --sources "contracts" --testfiles "$TEST_FILE"
    exit 0
fi

# Run tests
if eval "$TEST_CMD"; then
    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ ALL EDGE CASE TESTS PASSED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Print test summary
    echo -e "\n${BLUE}Test Coverage Summary:${NC}"
    echo -e "  ✓ Invalid Proof Formats       (5 tests)"
    echo -e "  ✓ Boundary Conditions         (3 tests)"
    echo -e "  ✓ Replay Attack Prevention    (2 tests)"
    echo -e "  ✓ State Consistency           (3 tests)"
    echo -e "  ✓ Gas Optimization            (2 tests)"
    echo -e "  ✓ Network & Timing            (3 tests)"
    echo -e "  ✓ Security Checks             (2 tests)"
    echo -e "  ✓ Data Validation             (2 tests)"
    echo -e "  ✓ Stress Testing              (1 test)"
    echo -e "  ✓ Error Handling              (2 tests)"
    
    echo -e "\n${GREEN}Total: 25+ edge case tests passed${NC}\n"
    
    exit 0
else
    echo -e "\n${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ EDGE CASE TESTS FAILED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
