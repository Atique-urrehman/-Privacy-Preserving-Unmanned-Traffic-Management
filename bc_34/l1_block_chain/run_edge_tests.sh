#!/bin/bash

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

echo -e "${BLUE}  Edge Case Testing Suite - BatchProof System${NC}"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED} Error: Test file not found at $TEST_FILE${NC}"
    exit 1
fi

echo -e "${GREEN} Test file found${NC}"
echo -e "${YELLOW}Running tests...${NC}\n"

# Build test command
TEST_CMD="npx hardhat test $TEST_FILE --network $NETWORK"

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
    echo -e "${GREEN}ALL EDGE CASE TESTS PASSED${NC}"
    
    # Print test summary
    echo -e "\n${BLUE}Test Coverage Summary:${NC}"
    echo -e "   Verifier field boundary     (1 test)"
    echo -e "   Verifier zero proof         (1 test)"
    echo -e "   Registry root update        (1 test)"
    echo -e "   Duplicate root submissions  (1 test)"
    echo -e "   Sequential submissions      (1 test)"

    echo -e "\n${GREEN}Total: 5 edge case tests passed${NC}\n"
    
    exit 0
else
    echo -e "${RED}EDGE CASE TESTS FAILED${NC}"
    exit 1
fi
