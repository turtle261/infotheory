#!/usr/bin/env python3
"""
Mathematical verification for information theory primitives.
Fixed version that handles your CLI output format and diagnoses issues.
"""

import numpy as np
import subprocess
import os
from pathlib import Path
import math
import re

TEST_DIR = Path("test_data_analytical")
TEST_DIR.mkdir(exist_ok=True)
BINARY = "./target/release/infotheory"

def entropy(probs):
    """Calculate Shannon entropy H(X) = -sum p(x) log2 p(x)"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))

def tvd(p, q):
    """Total Variation Distance"""
    return 0.5 * np.sum(np.abs(np.array(p) - np.array(q)))

def hellinger(p, q):
    """Normalized Hellinger Distance"""
    bc = np.sum(np.sqrt(np.array(p) * np.array(q)))  # Bhattacharyya coefficient
    return np.sqrt(1 - bc)

def create_distribution_file(probs, symbols, filename, count=10000):
    """Create a file representing a probability distribution"""
    probs = np.array(probs)
    probs = probs / np.sum(probs)  # Normalize
    
    # Ensure symbols are uint8 for proper byte output
    symbols = np.array(symbols, dtype=np.uint8)
    samples = np.random.choice(symbols, size=count, p=probs)
    with open(TEST_DIR / filename, 'wb') as f:
        f.write(samples.astype(np.uint8).tobytes())

def run_infotheory(primitive, file1, file2=None):
    """Run the infotheory binary and get result - FIXED PARSER"""
    if file2 is None:
        file2 = file1
    cmd = [BINARY, primitive, str(TEST_DIR / file1), str(TEST_DIR / file2)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout.strip()
    
    # Try to extract number from various formats
    # Format 1: "H(X) = 0.3033"
    match = re.search(r'=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', output)
    if match:
        return float(match.group(1))
    
    # Format 2: Just a number on last line
    try:
        return float(output.split('\n')[-1])
    except:
        print(f"❌ Could not parse output: '{output}'")
        print(f"   Command: {' '.join(cmd)}")
        return None

def test_case(name, expected, actual, tolerance=0.05):
    """Check if actual value matches expected within tolerance"""
    if actual is None:
        print(f"❌ {name}: FAILED TO RUN")
        return False
    
    diff = abs(actual - expected)
    if diff <= tolerance:
        print(f"✅ {name}")
        print(f"   Expected: {expected:.4f}, Got: {actual:.4f}, Diff: {diff:.4f}")
        return True
    else:
        print(f"❌ {name}")
        print(f"   Expected: {expected:.4f}, Got: {actual:.4f}, Diff: {diff:.4f}")
        return False

print("="*60)
print("DIAGNOSTIC MODE - Checking Basic Functionality")
print("="*60)

# Create simple test files
print("\n📝 Creating test files...")
create_distribution_file([0.5, 0.5], [ord('A'), ord('B')], "uniform_binary.bin", count=10000)
create_distribution_file([1.0, 0.0], [ord('A'), ord('B')], "all_a.bin", count=10000)
create_distribution_file([0.0, 1.0], [ord('A'), ord('B')], "all_b.bin", count=10000)

# DIAGNOSTIC 1: Check entropy output format and values
print("\n" + "="*60)
print("DIAGNOSTIC 1: Entropy Calculation")
print("="*60)

h_uniform = run_infotheory("entropy", "uniform_binary.bin")
print(f"\n📊 Uniform binary (50% A, 50% B):")
print(f"   Your output: {h_uniform:.4f}")
print(f"   Expected: ~1.0 bits/symbol (or ~0.125 bits/byte if per-byte)")
print(f"   Analysis: ", end="")

if h_uniform is None:
    print("❌ Failed to parse entropy output")
elif abs(h_uniform - 1.0) < 0.1:
    print("✅ Correct! Entropy per symbol")
elif abs(h_uniform - 0.125) < 0.05:
    print("⚠️  Looks like entropy per byte (bits/byte instead of bits/symbol)")
elif abs(h_uniform - 8.0) < 0.1:
    print("⚠️  Looks like bits per byte * 8? Unusual scaling")
else:
    print(f"❌ Unexpected value. Check your entropy calculation.")

# DIAGNOSTIC 2: Check if identical files give distance 0
print("\n" + "="*60)
print("DIAGNOSTIC 2: Identity Property (d(X,X) should be 0)")
print("="*60)

primitives_to_test = ["ned", "ned_cons", "nte", "tvd", "nhd"]
print(f"\n📊 Comparing uniform_binary.bin to itself:")

for prim in primitives_to_test:
    val = run_infotheory(prim, "uniform_binary.bin", "uniform_binary.bin")
    if val is not None:
        status = "✅" if abs(val) < 0.01 else "❌"
        print(f"   {status} {prim}: {val:.4f} (should be ~0.0)")
        
        if abs(val) > 0.5 and prim == "ned":
            print(f"      ⚠️  NED identity is failing badly!")
            print(f"      This suggests H(X,Y) != H(X) when comparing identical data")
            print(f"      Check your joint entropy calculation!")

# DIAGNOSTIC 3: Check TVD on completely different distributions
print("\n" + "="*60)
print("DIAGNOSTIC 3: TVD on Disjoint Distributions")
print("="*60)

print(f"\n📊 Comparing all_a.bin (100% A) vs all_b.bin (100% B):")
print(f"   Expected TVD: 1.0 (completely disjoint)")
print(f"   Expected NHD: 1.0 (completely disjoint)")

tvd_val = run_infotheory("tvd", "all_a.bin", "all_b.bin")
nhd_val = run_infotheory("nhd", "all_a.bin", "all_b.bin")

if tvd_val is not None:
    print(f"\n   Your TVD: {tvd_val:.4f}")
    if abs(tvd_val - 1.0) < 0.1:
        print(f"   ✅ Correct!")
    else:
        print(f"   ❌ Wrong! This suggests TVD is not comparing marginal distributions")
        print(f"   💡 Are you comparing context-conditional distributions instead?")
        print(f"   💡 TVD should compare p_X(symbol) vs p_Y(symbol), NOT contexts")

if nhd_val is not None:
    print(f"\n   Your NHD: {nhd_val:.4f}")
    if abs(nhd_val - 1.0) < 0.1:
        print(f"   ✅ Correct!")
    else:
        print(f"   ❌ Wrong! Same issue as TVD")

# DIAGNOSTIC 4: Mutual Information check
print("\n" + "="*60)
print("DIAGNOSTIC 4: Mutual Information I(X;X) = H(X)")
print("="*60)

h_x = run_infotheory("entropy", "uniform_binary.bin")
mi_xx = run_infotheory("mi", "uniform_binary.bin", "uniform_binary.bin")

if h_x is not None and mi_xx is not None:
    print(f"\n📊 For uniform_binary.bin:")
    print(f"   H(X): {h_x:.4f}")
    print(f"   I(X;X): {mi_xx:.4f}")
    print(f"   Difference: {abs(h_x - mi_xx):.4f}")
    
    if abs(h_x - mi_xx) < 0.01:
        print(f"   ✅ I(X;X) = H(X) holds!")
    else:
        print(f"   ❌ I(X;X) should equal H(X)!")

# DIAGNOSTIC 5: Joint entropy relationship
print("\n" + "="*60)
print("DIAGNOSTIC 5: Joint Entropy H(X,Y) Relationship")
print("="*60)

h_xy_same = run_infotheory("joint_entropy", "uniform_binary.bin", "uniform_binary.bin")

if h_x is not None and h_xy_same is not None:
    print(f"\n📊 For identical files:")
    print(f"   H(X): {h_x:.4f}")
    print(f"   H(X,Y) when Y=X: {h_xy_same:.4f}")
    print(f"   Difference: {abs(h_x - h_xy_same):.4f}")
    
    if abs(h_x - h_xy_same) < 0.01:
        print(f"   ✅ H(X,X) = H(X) holds!")
    else:
        print(f"   ❌ When X=Y, H(X,Y) should equal H(X)!")
        print(f"   💡 Check how you're computing joint entropy")
        print(f"   💡 Are you concatenating files? That would give H≈2*H(X)")

# Summary
print("\n" + "="*60)
print("SUMMARY OF ISSUES")
print("="*60)

issues = []

if h_uniform is not None and abs(h_uniform - 1.0) > 0.2:
    issues.append("⚠️  Entropy values seem off - check units (bits/symbol vs bits/byte)")

ned_identity = run_infotheory("ned", "uniform_binary.bin", "uniform_binary.bin")
if ned_identity is not None and abs(ned_identity) > 0.1:
    issues.append("❌ NED identity failing - check joint entropy calculation")

if tvd_val is not None and abs(tvd_val - 1.0) > 0.2:
    issues.append("❌ TVD comparing wrong distributions - should use marginals, not contexts")

if nhd_val is not None and abs(nhd_val - 1.0) > 0.2:
    issues.append("❌ NHD comparing wrong distributions - should use marginals, not contexts")

if h_xy_same is not None and h_x is not None and abs(h_xy_same - h_x) > 0.1:
    issues.append("❌ H(X,X) ≠ H(X) - joint entropy calculation is wrong")

if not issues:
    print("\n✅ All diagnostics passed! Your implementation looks correct.")
else:
    print("\nFound the following issues:")
    for issue in issues:
        print(f"  {issue}")
    
    print("\n" + "="*60)
    print("RECOMMENDED FIXES")
    print("="*60)
    print("""
1. For TVD and NHD:
   - These should compare MARGINAL distributions p_X(symbol) vs p_Y(symbol)
   - NOT context-conditional distributions from ROSA
   - You need to extract symbol frequencies, not use ROSA contexts
   
2. For NED and joint entropy:
   - H(X,Y) when X=Y should equal H(X)
   - If you're concatenating files [X,Y], that's wrong
   - You need to build a joint probability model p(x,y)
   
3. For entropy normalization:
   - Decide: bits per symbol or bits per byte?
   - For binary alphabet: max entropy is 1 bit/symbol
   - For 256-byte alphabet: max entropy is 8 bits/byte
    """)

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("""
1. Run this diagnostic again after fixes
2. Check your ROSA implementation for how it handles:
   - Marginal probability extraction
   - Joint probability modeling
3. Look at your spec.md formulas - are you implementing them exactly?
""")
