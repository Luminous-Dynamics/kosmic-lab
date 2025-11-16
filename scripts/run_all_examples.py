#!/usr/bin/env python3
"""
Run All Examples - Execute all Kosmic Lab examples and generate summary

This script:
1. Discovers all example files (examples/*.py)
2. Runs each example with timeout
3. Captures output and errors
4. Generates summary report
5. Validates outputs exist

Usage:
    python scripts/run_all_examples.py

    or

    make run-examples

Options:
    --quick: Skip slow examples (>30 seconds)
    --verbose: Show full output for each example
    --stop-on-error: Stop if any example fails

Author: Kosmic Lab Team
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def discover_examples(quick: bool = False) -> List[Path]:
    """
    Discover all example files.

    Args:
        quick: If True, skip slow examples

    Returns:
        List of example file paths
    """
    examples_dir = Path("examples")

    if not examples_dir.exists():
        print(f"{Colors.RED}Error: examples/ directory not found{Colors.RESET}")
        sys.exit(1)

    # Find all Python files except __init__.py and interactive tutorial
    examples = sorted(
        examples_dir.glob("*.py"),
        key=lambda p: p.name
    )

    # Filter out non-example files
    examples = [
        ex for ex in examples
        if ex.name not in ["__init__.py", "interactive_tutorial.py"]
    ]

    # Skip slow examples if quick mode
    if quick:
        # Examples 03 and 04 can be slow
        examples = [ex for ex in examples if ex.name not in ["03_multi_universe.py"]]

    return examples


def run_example(example_path: Path, timeout: int = 120, verbose: bool = False) -> Dict:
    """
    Run a single example and capture results.

    Args:
        example_path: Path to example file
        timeout: Timeout in seconds
        verbose: Show full output

    Returns:
        Dictionary with results
    """
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Running: {example_path.name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")

    result = {
        "name": example_path.name,
        "path": str(example_path),
        "success": False,
        "runtime": 0.0,
        "output": "",
        "error": "",
    }

    start_time = time.time()

    try:
        # Run example with PYTHONPATH set
        env = {"PYTHONPATH": str(Path.cwd())}

        process = subprocess.run(
            ["poetry", "run", "python", str(example_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**subprocess.os.environ, **env}
        )

        result["runtime"] = time.time() - start_time
        result["output"] = process.stdout
        result["error"] = process.stderr
        result["success"] = (process.returncode == 0)

        if result["success"]:
            print(f"{Colors.GREEN}✓ SUCCESS{Colors.RESET} ({result['runtime']:.1f}s)")
        else:
            print(f"{Colors.RED}✗ FAILED{Colors.RESET} (exit code: {process.returncode})")
            if verbose:
                print(f"\n{Colors.RED}Error output:{Colors.RESET}")
                print(process.stderr[:500])  # First 500 chars

    except subprocess.TimeoutExpired:
        result["runtime"] = timeout
        result["error"] = f"Timeout after {timeout}s"
        print(f"{Colors.RED}✗ TIMEOUT{Colors.RESET} (>{timeout}s)")

    except Exception as e:
        result["runtime"] = time.time() - start_time
        result["error"] = str(e)
        print(f"{Colors.RED}✗ ERROR{Colors.RESET}: {e}")

    if verbose and result["output"]:
        print(f"\n{Colors.YELLOW}Output preview:{Colors.RESET}")
        print(result["output"][:500])  # First 500 chars

    return result


def validate_outputs(results: List[Dict]) -> Dict[str, List[str]]:
    """
    Validate that examples produced expected outputs.

    Args:
        results: List of example results

    Returns:
        Dictionary with validation results
    """
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Validating Outputs{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

    validation = {
        "logs_created": [],
        "outputs_created": [],
        "kcodex_created": [],
        "missing": [],
    }

    # Check common output locations
    logs_dir = Path("logs")
    outputs_dir = Path("outputs")

    if logs_dir.exists():
        log_files = list(logs_dir.glob("**/*.log")) + list(logs_dir.glob("**/*.json"))
        validation["logs_created"] = [str(f) for f in log_files]
        print(f"{Colors.GREEN}✓{Colors.RESET} Found {len(log_files)} log files")

    if outputs_dir.exists():
        output_files = list(outputs_dir.glob("**/*"))
        output_files = [f for f in output_files if f.is_file()]
        validation["outputs_created"] = [str(f) for f in output_files]
        print(f"{Colors.GREEN}✓{Colors.RESET} Found {len(output_files)} output files")

    # Check for K-Codex files
    kcodex_files = []
    if logs_dir.exists():
        kcodex_files = list(logs_dir.glob("**/*kcodex.json"))
        validation["kcodex_created"] = [str(f) for f in kcodex_files]
        print(f"{Colors.GREEN}✓{Colors.RESET} Found {len(kcodex_files)} K-Codex files")

    return validation


def print_summary(results: List[Dict], validation: Dict):
    """
    Print summary report.

    Args:
        results: List of example results
        validation: Validation results
    """
    print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY REPORT{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    total_time = sum(r["runtime"] for r in results)

    print(f"📊 {Colors.BOLD}Examples Run:{Colors.RESET} {total}")
    print(f"✅ {Colors.GREEN}Passed:{Colors.RESET} {passed}")
    print(f"❌ {Colors.RED}Failed:{Colors.RESET} {failed}")
    print(f"⏱️  {Colors.BOLD}Total Time:{Colors.RESET} {total_time:.1f}s")

    if failed > 0:
        print(f"\n{Colors.RED}Failed Examples:{Colors.RESET}")
        for r in results:
            if not r["success"]:
                print(f"  • {r['name']}: {r['error'][:100]}")

    print(f"\n📂 {Colors.BOLD}Outputs:{Colors.RESET}")
    print(f"  • Log files: {len(validation['logs_created'])}")
    print(f"  • Output files: {len(validation['outputs_created'])}")
    print(f"  • K-Codex files: {len(validation['kcodex_created'])}")

    print(f"\n⚡ {Colors.BOLD}Performance:{Colors.RESET}")
    for r in sorted(results, key=lambda x: x["runtime"], reverse=True):
        status = f"{Colors.GREEN}✓{Colors.RESET}" if r["success"] else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {r['name']:<35} {r['runtime']:>6.1f}s")

    # Overall status
    print(f"\n{Colors.BOLD}Overall Status:{Colors.RESET}")
    if failed == 0:
        print(f"{Colors.GREEN}{'='*70}")
        print(f"✅ ALL EXAMPLES PASSED!")
        print(f"{'='*70}{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}{'='*70}")
        print(f"❌ {failed}/{total} EXAMPLES FAILED")
        print(f"{'='*70}{Colors.RESET}\n")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Run all Kosmic Lab examples")
    parser.add_argument("--quick", action="store_true", help="Skip slow examples")
    parser.add_argument("--verbose", action="store_true", help="Show full output")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop if example fails")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per example (seconds)")

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"🌊 KOSMIC LAB - RUNNING ALL EXAMPLES")
    print(f"{'='*70}{Colors.RESET}\n")

    # Discover examples
    examples = discover_examples(quick=args.quick)
    print(f"Found {len(examples)} examples to run\n")

    if args.quick:
        print(f"{Colors.YELLOW}Quick mode: Skipping slow examples{Colors.RESET}\n")

    # Run all examples
    results = []
    for example in examples:
        result = run_example(example, timeout=args.timeout, verbose=args.verbose)
        results.append(result)

        if not result["success"] and args.stop_on_error:
            print(f"\n{Colors.RED}Stopping due to error{Colors.RESET}")
            break

    # Validate outputs
    validation = validate_outputs(results)

    # Print summary
    print_summary(results, validation)

    # Exit code
    failed = sum(1 for r in results if not r["success"])
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
