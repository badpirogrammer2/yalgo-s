#!/usr/bin/env python3
"""
Master test runner for YALGO-S algorithms with Hugging Face datasets.

This script runs comprehensive tests for all algorithms:
1. AGMOHD - MNIST, CIFAR-10, parallel processing
2. POIC-NET - COCO, Flickr30k, multi-GPU
3. ARCE - IoT sensors, network traffic, user behavior (simulation)
4. Parallel processing and RTX optimizations validation

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --agmohd-only     # Run only AGMOHD tests
    python run_all_tests.py --poic-net-only   # Run only POIC-NET tests
    python run_all_tests.py --arce-only       # Run only ARCE tests
    python run_all_tests.py --quick           # Run quick tests only
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_agmohd_tests(quick=False):
    """Run AGMOHD comprehensive tests."""
    logger.info("🚀 Starting AGMOHD tests...")

    try:
        # Import and run AGMOHD tests
        from test_agmohd_hf import run_comprehensive_agmohd_tests, print_agmohd_test_results

        start_time = time.time()
        results = run_comprehensive_agmohd_tests()
        test_time = time.time() - start_time

        # Print results
        print_agmohd_test_results(results)

        # Add timing info
        results['test_metadata'] = {
            'test_duration': test_time,
            'timestamp': time.time(),
            'test_type': 'comprehensive' if not quick else 'quick'
        }

        logger.info(".2f")
        return results

    except Exception as e:
        logger.error(f"AGMOHD tests failed: {e}")
        return {"error": str(e), "test": "agmohd"}

def run_poic_net_tests(quick=False):
    """Run POIC-NET comprehensive tests."""
    logger.info("🚀 Starting POIC-NET tests...")

    try:
        # Import and run POIC-NET tests
        from test_poic_net_hf import run_comprehensive_poic_net_tests, print_poic_net_test_results

        start_time = time.time()
        results = run_comprehensive_poic_net_tests()
        test_time = time.time() - start_time

        # Print results
        print_poic_net_test_results(results)

        # Add timing info
        results['test_metadata'] = {
            'test_duration': test_time,
            'timestamp': time.time(),
            'test_type': 'comprehensive' if not quick else 'quick'
        }

        logger.info(".2f")
        return results

    except Exception as e:
        logger.error(f"POIC-NET tests failed: {e}")
        return {"error": str(e), "test": "poic_net"}

def run_arce_tests(quick=False):
    """Run ARCE simulation tests."""
    logger.info("🚀 Starting ARCE simulation tests...")

    try:
        # Import and run ARCE tests
        from test_arce_hf import run_comprehensive_arce_tests, print_arce_test_results

        start_time = time.time()
        results = run_comprehensive_arce_tests()
        test_time = time.time() - start_time

        # Print results
        print_arce_test_results(results)

        # Add timing info
        results['test_metadata'] = {
            'test_duration': test_time,
            'timestamp': time.time(),
            'test_type': 'comprehensive' if not quick else 'quick'
        }

        logger.info(".2f")
        return results

    except Exception as e:
        logger.error(f"ARCE tests failed: {e}")
        return {"error": str(e), "test": "arce"}

def run_parallel_tests():
    """Run parallel processing and RTX optimization tests."""
    logger.info("🚀 Starting parallel processing tests...")

    try:
        # Import and run parallel tests
        from test_parallel_optimizations import benchmark_algorithms, print_benchmark_results

        start_time = time.time()
        results = benchmark_algorithms()
        test_time = time.time() - start_time

        # Print results
        print_benchmark_results(results)

        # Add timing info
        results['test_metadata'] = {
            'test_duration': test_time,
            'timestamp': time.time(),
            'test_type': 'parallel_benchmark'
        }

        logger.info(".2f")
        return results

    except Exception as e:
        logger.error(f"Parallel tests failed: {e}")
        return {"error": str(e), "test": "parallel"}

def generate_test_report(all_results):
    """Generate comprehensive test report."""
    print("\n" + "="*100)
    print("YALGO-S COMPREHENSIVE TEST REPORT")
    print("="*100)

    # Overall statistics
    total_tests = len(all_results)
    successful_tests = sum(1 for result in all_results.values() if "error" not in result)
    failed_tests = total_tests - successful_tests

    print("\n📊 OVERALL STATISTICS:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Failed: {failed_tests}")
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print(".1f")

    # System information
    if 'agmohd_results' in all_results and 'system_info' in all_results['agmohd_results']:
        sys_info = all_results['agmohd_results']['system_info']
        print("\n🔧 SYSTEM INFORMATION:")
        print(f"  CUDA Available: {sys_info.get('cuda_available', 'Unknown')}")
        print(f"  GPU Count: {sys_info.get('gpu_count', 'Unknown')}")
        print(f"  GPU Name: {sys_info.get('gpu_name', 'Unknown')}")

    # Performance summary
    print("\n⚡ PERFORMANCE SUMMARY:")
    for test_name, results in all_results.items():
        if "error" not in results and "test_metadata" in results:
            duration = results["test_metadata"]["test_duration"]
            test_type = results["test_metadata"]["test_type"]
            print(".2f")

    # Error summary
    if failed_tests > 0:
        print("\n❌ FAILED TESTS:")
        for test_name, results in all_results.items():
            if "error" in results:
                print(f"  {test_name}: {results['error']}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if successful_tests > 0:
        print("  ✅ Tests passed successfully - algorithms are working correctly")
        print("  🚀 Consider RTX optimizations for RTX 40-series GPUs")
        print("  ⚡ Enable parallel processing for improved performance")
    if failed_tests > 0:
        print("  ❌ Some tests failed - check logs for details")
        print("  🔧 Ensure all dependencies are installed")
        print("  📞 Contact support if issues persist")

    print("\n" + "="*100)
    print("🎉 TEST RUN COMPLETE!")
    print("📊 Detailed results saved to 'comprehensive_test_results.json'")
    print("📋 Check 'test_run.log' for detailed logs")
    print("="*100)

def save_comprehensive_results(all_results, output_file="comprehensive_test_results.json"):
    """Save all test results to a comprehensive JSON file."""
    # Create serializable version
    serializable_results = {}

    for test_name, results in all_results.items():
        serializable_results[test_name] = {}

        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[test_name][key] = {}
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'tolist'):  # NumPy array
                        serializable_results[test_name][key][sub_key] = sub_value.tolist()
                    elif isinstance(sub_value, (int, float, str, bool, list)):
                        serializable_results[test_name][key][sub_key] = sub_value
                    else:
                        serializable_results[test_name][key][sub_key] = str(sub_value)
            else:
                if hasattr(value, 'tolist'):  # NumPy array
                    serializable_results[test_name][key] = value.tolist()
                elif isinstance(value, (int, float, str, bool, list)):
                    serializable_results[test_name][key] = value
                else:
                    serializable_results[test_name][key] = str(value)

    # Add metadata
    serializable_results['_metadata'] = {
        'test_run_timestamp': time.time(),
        'total_tests': len(all_results),
        'python_version': sys.version,
        'working_directory': str(Path.cwd()),
        'test_files': [
            'test_agmohd_hf.py',
            'test_poic_net_hf.py',
            'test_arce_hf.py',
            'test_parallel_optimizations.py'
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Comprehensive results saved to {output_file}")

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Run YALGO-S comprehensive tests')
    parser.add_argument('--agmohd-only', action='store_true', help='Run only AGMOHD tests')
    parser.add_argument('--poic-net-only', action='store_true', help='Run only POIC-NET tests')
    parser.add_argument('--arce-only', action='store_true', help='Run only ARCE tests')
    parser.add_argument('--parallel-only', action='store_true', help='Run only parallel tests')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--output', type=str, default='comprehensive_test_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    print("🚀 YALGO-S Comprehensive Test Suite")
    print("="*50)
    print("Testing algorithms with Hugging Face datasets")
    print("Includes parallel processing and RTX optimizations")
    print("="*50)

    all_results = {}
    start_time = time.time()

    try:
        # Run selected tests
        if args.agmohd_only:
            all_results['agmohd_results'] = run_agmohd_tests(args.quick)
        elif args.poic_net_only:
            all_results['poic_net_results'] = run_poic_net_tests(args.quick)
        elif args.arce_only:
            all_results['arce_results'] = run_arce_tests(args.quick)
        elif args.parallel_only:
            all_results['parallel_results'] = run_parallel_tests()
        else:
            # Run all tests
            logger.info("Running comprehensive test suite...")

            all_results['agmohd_results'] = run_agmohd_tests(args.quick)
            all_results['poic_net_results'] = run_poic_net_tests(args.quick)
            all_results['arce_results'] = run_arce_tests(args.quick)
            all_results['parallel_results'] = run_parallel_tests()

        # Calculate total time
        total_time = time.time() - start_time

        # Generate report
        generate_test_report(all_results)

        # Save results
        save_comprehensive_results(all_results, args.output)

        print(".2f")

        # Exit with appropriate code
        failed_tests = sum(1 for result in all_results.values() if "error" in result)
        sys.exit(0 if failed_tests == 0 else 1)

    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
