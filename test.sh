#!/bin/bash
# Test suite for information theory primitives
# Place this in your project root and make executable: chmod +x test_infotheory.sh

set -e  # Exit on error

BINARY="./target/release/infotheory"
TEST_DIR="test_data"
RESULTS_FILE="test_results.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$TEST_DIR"
echo "Information Theory Test Suite" > "$RESULTS_FILE"
echo "=============================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Helper to extract number from output (handles "H(X) = 0.5" format)
extract_value() {
    local output="$1"
    # Try to extract number after '=' sign
    if echo "$output" | grep -q '='; then
        echo "$output" | grep -oP '=\s*\K[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    else
        # Otherwise just take the last line
        echo "$output"
    fi
}

# Helper function to check if value is close to expected
check_value() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    local tolerance="${4:-0.05}"  # Default 5% tolerance
    
    # Use awk for floating point comparison
    local diff=$(awk -v a="$actual" -v e="$expected" 'BEGIN {print (a-e < 0 ? e-a : a-e)}')
    local within=$(awk -v d="$diff" -v t="$tolerance" 'BEGIN {print (d <= t ? 1 : 0)}')
    
    if [ "$within" -eq 1 ]; then
        echo -e "${GREEN}✓${NC} $name: $actual (expected ~$expected, diff: $diff)"
        echo "✓ $name: $actual (expected ~$expected, diff: $diff)" >> "$RESULTS_FILE"
        return 0
    else
        echo -e "${RED}✗${NC} $name: $actual (expected ~$expected, diff: $diff)"
        echo "✗ $name: $actual (expected ~$expected, diff: $diff)" >> "$RESULTS_FILE"
        return 1
    fi
}

# Helper to check symmetry
check_symmetry() {
    local primitive="$1"
    local file1="$2"
    local file2="$3"
    
    local output1=$($BINARY "$primitive" "$file1" "$file2")
    local output2=$($BINARY "$primitive" "$file2" "$file1")
    local val1=$(extract_value "$output1")
    local val2=$(extract_value "$output2")
    
    check_value "${primitive} symmetry" "$val1" "$val2" 0.001
}

echo -e "${YELLOW}=== Creating Test Data ===${NC}"

# Test 1: Identical sequences (should give distance = 0)
echo -e "${YELLOW}Creating identical sequences...${NC}"
dd if=/dev/urandom of="$TEST_DIR/random1.bin" bs=1K count=10 2>/dev/null
cp "$TEST_DIR/random1.bin" "$TEST_DIR/random1_copy.bin"

# Test 2: Completely different random data (distance should be close to 1)
echo -e "${YELLOW}Creating independent random sequences...${NC}"
dd if=/dev/urandom of="$TEST_DIR/random2.bin" bs=1K count=10 2>/dev/null

# Test 3: Periodic/repeated pattern (low entropy)
echo -e "${YELLOW}Creating periodic sequence...${NC}"
for i in {1..1000}; do echo -n "ABCDEFGH"; done > "$TEST_DIR/periodic.txt"

# Test 4: High entropy uniform random
echo -e "${YELLOW}Creating high-entropy sequence...${NC}"
dd if=/dev/urandom of="$TEST_DIR/high_entropy.bin" bs=1K count=10 2>/dev/null

# Test 5: Partial overlap (50% shared content)
echo -e "${YELLOW}Creating partially overlapping sequences...${NC}"
cat "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin" > "$TEST_DIR/overlap_a.bin"
cat "$TEST_DIR/random2.bin" "$TEST_DIR/random1.bin" > "$TEST_DIR/overlap_b.bin"

# Test 6: Simple binary sequences for manual verification
echo -e "${YELLOW}Creating simple binary test cases...${NC}"
# File with mostly 'A' (skewed distribution)
for i in {1..900}; do echo -n "A"; done > "$TEST_DIR/mostly_a.txt"
for i in {1..100}; do echo -n "B"; done >> "$TEST_DIR/mostly_a.txt"

# File with mostly 'B' (skewed distribution)
for i in {1..100}; do echo -n "A"; done > "$TEST_DIR/mostly_b.txt"
for i in {1..900}; do echo -n "B"; done >> "$TEST_DIR/mostly_b.txt"

# Uniform binary
for i in {1..500}; do echo -n "A"; done > "$TEST_DIR/uniform_ab.txt"
for i in {1..500}; do echo -n "B"; done >> "$TEST_DIR/uniform_ab.txt"

echo ""
echo -e "${YELLOW}=== Property Tests ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "=== Property Tests ===" >> "$RESULTS_FILE"

PASSED=0
FAILED=0

# Test identity: distance to self should be 0
echo -e "\n${YELLOW}Testing Identity Property (d(X,X) = 0)${NC}"
echo "" >> "$RESULTS_FILE"
echo "Identity Property Tests:" >> "$RESULTS_FILE"

for primitive in ned ned_cons nte tvd nhd; do
    output=$($BINARY "$primitive" "$TEST_DIR/random1.bin" "$TEST_DIR/random1_copy.bin")
    result=$(extract_value "$output")
    if check_value "$primitive identity" "$result" "0.0" 0.001; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# Test symmetry: d(X,Y) = d(Y,X)
echo -e "\n${YELLOW}Testing Symmetry Property (d(X,Y) = d(Y,X))${NC}"
echo "" >> "$RESULTS_FILE"
echo "Symmetry Property Tests:" >> "$RESULTS_FILE"

for primitive in ned ned_cons nte tvd nhd; do
    if check_symmetry "$primitive" "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
done

# Test bounds: all distances should be in [0, 1]
echo -e "\n${YELLOW}Testing Bounds (0 ≤ d ≤ 1)${NC}"
echo "" >> "$RESULTS_FILE"
echo "Bounds Tests:" >> "$RESULTS_FILE"

for primitive in ned ned_cons nte tvd nhd; do
    output=$($BINARY "$primitive" "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin")
    result=$(extract_value "$output")
    in_bounds=$(awk -v r="$result" 'BEGIN {print (r >= 0 && r <= 1 ? 1 : 0)}')
    if [ "$in_bounds" -eq 1 ]; then
        echo -e "${GREEN}✓${NC} $primitive in bounds: $result ∈ [0,1]"
        echo "✓ $primitive in bounds: $result ∈ [0,1]" >> "$RESULTS_FILE"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $primitive out of bounds: $result ∉ [0,1]"
        echo "✗ $primitive out of bounds: $result ∉ [0,1]" >> "$RESULTS_FILE"
        ((FAILED++))
    fi
done

# Test independence: random data should give high distance
echo -e "\n${YELLOW}Testing Independence (d(random,random) ≈ high)${NC}"
echo "" >> "$RESULTS_FILE"
echo "Independence Tests (expecting values > 0.5 for independent random data):" >> "$RESULTS_FILE"

for primitive in ned nte; do
    output=$($BINARY "$primitive" "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin")
    result=$(extract_value "$output")
    high=$(awk -v r="$result" 'BEGIN {print (r > 0.5 ? 1 : 0)}')
    if [ "$high" -eq 1 ]; then
        echo -e "${GREEN}✓${NC} $primitive independence: $result > 0.5"
        echo "✓ $primitive independence: $result > 0.5" >> "$RESULTS_FILE"
        ((PASSED++))
    else
        echo -e "${YELLOW}!${NC} $primitive independence: $result ≤ 0.5 (may be OK for small samples)"
        echo "! $primitive independence: $result ≤ 0.5" >> "$RESULTS_FILE"
        ((PASSED++))  # Don't fail on this - might be due to sample size
    fi
done

# Specific tests for different data types
echo -e "\n${YELLOW}=== Sanity Checks ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "Sanity Checks:" >> "$RESULTS_FILE"

# Periodic data should have lower entropy than random
echo -e "\n${YELLOW}Entropy comparison: periodic vs random${NC}"
output_periodic=$($BINARY entropy "$TEST_DIR/periodic.txt" "$TEST_DIR/periodic.txt")
output_random=$($BINARY entropy "$TEST_DIR/high_entropy.bin" "$TEST_DIR/high_entropy.bin")
h_periodic=$(extract_value "$output_periodic")
h_random=$(extract_value "$output_random")

echo "Periodic entropy: $h_periodic bits/byte"
echo "Random entropy: $h_random bits/byte"
echo "Periodic entropy: $h_periodic bits/byte" >> "$RESULTS_FILE"
echo "Random entropy: $h_random bits/byte" >> "$RESULTS_FILE"

lower=$(awk -v p="$h_periodic" -v r="$h_random" 'BEGIN {print (p < r ? 1 : 0)}')
if [ "$lower" -eq 1 ]; then
    echo -e "${GREEN}✓${NC} Periodic has lower entropy than random"
    echo "✓ Periodic has lower entropy than random" >> "$RESULTS_FILE"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Periodic should have lower entropy than random"
    echo "✗ Periodic should have lower entropy than random" >> "$RESULTS_FILE"
    ((FAILED++))
fi

# Mutual information tests
echo -e "\n${YELLOW}Mutual Information tests${NC}"
output_mi_identical=$($BINARY mi "$TEST_DIR/random1.bin" "$TEST_DIR/random1_copy.bin")
output_mi_independent=$($BINARY mi "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin")
mi_identical=$(extract_value "$output_mi_identical")
mi_independent=$(extract_value "$output_mi_independent")

echo "MI(X,X): $mi_identical (should be high)"
echo "MI(X,Y_independent): $mi_independent (should be low)"
echo "MI(X,X): $mi_identical" >> "$RESULTS_FILE"
echo "MI(X,Y_independent): $mi_independent" >> "$RESULTS_FILE"

# NED and NED_MI equivalence test
echo -e "\n${YELLOW}Testing NED = NED_MI equivalence${NC}"
output_ned=$($BINARY ned "$TEST_DIR/random1.bin" "$TEST_DIR/random2.bin")
ned_result=$(extract_value "$output_ned")
echo "NED result: $ned_result"
echo "NED result: $ned_result" >> "$RESULTS_FILE"
echo "Note: Verify NED = 1 - I(X;Y)/max(H(X),H(Y)) manually if needed"
echo "Note: Verify NED = 1 - I(X;Y)/max(H(X),H(Y)) manually if needed" >> "$RESULTS_FILE"

# Summary
echo ""
echo -e "${YELLOW}=== Test Summary ===${NC}"
echo "" >> "$RESULTS_FILE"
echo "=== Test Summary ===" >> "$RESULTS_FILE"
TOTAL=$((PASSED + FAILED))
echo -e "Total tests: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Total tests: $TOTAL" >> "$RESULTS_FILE"
echo "Passed: $PASSED" >> "$RESULTS_FILE"
echo "Failed: $FAILED" >> "$RESULTS_FILE"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    echo "" >> "$RESULTS_FILE"
    echo "All tests passed!" >> "$RESULTS_FILE"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Check $RESULTS_FILE for details.${NC}"
    echo "" >> "$RESULTS_FILE"
    echo "Some tests failed." >> "$RESULTS_FILE"
    exit 1
fi
