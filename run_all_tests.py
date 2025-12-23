#!/usr/bin/env python3
"""
Main script to run all Motivation Tests for FusedANNS
"""
import os
import sys
import argparse
from pathlib import Path

# Add motivation_tests to path
sys.path.insert(0, str(Path(__file__).parent))

from motivation_tests.test_m1 import run_test_m1
from motivation_tests.test_m2 import run_test_m2
from motivation_tests.test_m3 import run_test_m3
from motivation_tests.test_m4 import run_test_m4
from motivation_tests.test_m5 import run_test_m5
from motivation_tests.test_m6 import run_test_m6


def main():
    parser = argparse.ArgumentParser(description="Run FusedANNS Motivation Tests")
    parser.add_argument("--test", type=int, choices=[1, 2, 3, 4, 5, 6], 
                       help="Run specific test (1-6), or omit to run all")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Base output directory for results")
    args = parser.parse_args()
    
    tests = {
        1: ("One-modality-one-vector vs Multi-vector", run_test_m1),
        2: ("Vespa/Lance Multi-vector vs MSTM-aware", run_test_m2),
        3: ("MUST Fused Vector vs Unified Similarity", run_test_m3),
        4: ("α-Reachable / Hausdorff Theory Validation", run_test_m4),
        5: ("Performance: Object-node vs Vector-node", run_test_m5),
        6: ("System-level: Layout + GPU Batch", run_test_m6),
    }
    
    if args.test:
        # Run specific test
        test_num = args.test
        if test_num not in tests:
            print(f"Error: Test {test_num} not found")
            return 1
        
        test_name, test_func = tests[test_num]
        print(f"\n{'='*80}")
        print(f"Running Test M{test_num}: {test_name}")
        print(f"{'='*80}\n")
        
        try:
            output_dir = os.path.join(args.output_dir, f"m{test_num}")
            test_func(output_dir=output_dir)
            print(f"\n✓ Test M{test_num} completed successfully!")
            return 0
        except Exception as e:
            print(f"\n✗ Test M{test_num} failed with error:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        # Run all tests sequentially
        print(f"\n{'='*80}")
        print("Running All Motivation Tests for FusedANNS")
        print(f"{'='*80}\n")
        
        results = {}
        for test_num in sorted(tests.keys()):
            test_name, test_func = tests[test_num]
            print(f"\n{'='*80}")
            print(f"Test M{test_num}/6: {test_name}")
            print(f"{'='*80}\n")
            
            try:
                output_dir = os.path.join(args.output_dir, f"m{test_num}")
                test_func(output_dir=output_dir)
                results[test_num] = "PASSED"
                print(f"\n✓ Test M{test_num} completed successfully!")
            except Exception as e:
                results[test_num] = f"FAILED: {str(e)}"
                print(f"\n✗ Test M{test_num} failed with error:")
                print(f"  {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*80}")
        print("Test Summary")
        print(f"{'='*80}")
        for test_num in sorted(tests.keys()):
            status = results[test_num]
            symbol = "✓" if status == "PASSED" else "✗"
            print(f"  {symbol} Test M{test_num}: {tests[test_num][0]} - {status}")
        print(f"{'='*80}\n")
        
        # Return 0 if all passed, 1 otherwise
        return 0 if all(r == "PASSED" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

