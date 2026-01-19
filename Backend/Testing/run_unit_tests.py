#!/usr/bin/env python3
"""
Unit Test Runner
Runs all unit tests and provides summary
"""

import unittest
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def discover_and_run_tests():
    """Discover and run all unit tests"""
    
    print("=" * 60)
    print("🧪 TRADING AGENT UNIT TESTS")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Discover tests in unit_tests directory
    test_dir = os.path.join(os.path.dirname(__file__), 'unit_tests')
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n🔍 FAILURES:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback}")
    
    if result.errors:
        print("\n🔍 ERRORS:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback}")
    
    # Return appropriate exit code
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED!")
        return 1


if __name__ == '__main__':
    # Run tests and exit with appropriate code
    exit_code = discover_and_run_tests()
    sys.exit(exit_code)