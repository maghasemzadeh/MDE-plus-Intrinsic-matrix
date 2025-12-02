"""
Test runner script with nice output formatting.
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run all tests with verbose output."""
    print("=" * 80)
    print("DEPTH ESTIMATION EVALUATION - TEST SUITE")
    print("=" * 80)
    print()
    
    # Get test directory
    test_dir = Path(__file__).parent
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        str(test_dir),
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        "-ra",  # Show all test result summary
        "--durations=10",  # Show 10 slowest tests
    ])
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ SOME TESTS FAILED (exit code: {exit_code})")
    print("=" * 80)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

