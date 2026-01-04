import time
import sys
import tracemalloc
import psutil
import statistics
import json
import argparse
from typing import Dict, List
from pathlib import Path


def profile_startup(iterations: int = 10) -> Dict:
    print(f"Profiling startup performance ({iterations} iterations)...")
    
    startup_times = []
    memory_peaks = []
    
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}...", end='\r')
        
        tracemalloc.start()
        process = psutil.Process()
        start_rss = process.memory_info().rss
        
        start = time.time()
        
        # Import main application (simulate startup)
        # Remove from cache to force reimport
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith('api') or k == 'main']
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Import main to measure startup
        try:
            import main
        except Exception as e:
            print(f"\nWarning: Could not import main module: {e}")
            # Continue profiling even if import fails
        
        elapsed = time.time() - start
        startup_times.append(elapsed * 1000)  # Convert to ms
        
        # Memory metrics
        current, peak = tracemalloc.get_traced_memory()
        end_rss = process.memory_info().rss
        tracemalloc.stop()
        
        memory_peaks.append({
            'python_mb': peak / 1024 / 1024,
            'rss_mb': (end_rss - start_rss) / 1024 / 1024
        })
    
    print()  # New line after iterations
    
    return {
        'iterations': iterations,
        'startup_ms_avg': statistics.mean(startup_times),
        'startup_ms_median': statistics.median(startup_times),
        'startup_ms_stdev': statistics.stdev(startup_times) if len(startup_times) > 1 else 0,
        'startup_ms_min': min(startup_times),
        'startup_ms_max': max(startup_times),
        'memory_python_mb_avg': statistics.mean([m['python_mb'] for m in memory_peaks]),
        'memory_rss_mb_avg': statistics.mean([m['rss_mb'] for m in memory_peaks])
    }


def print_results(results: Dict, label: str = "Results"):
    """Pretty print profiling results"""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Iterations:          {results['iterations']}")
    print(f"\nStartup Time:")
    print(f"  Average:           {results['startup_ms_avg']:.1f} ms")
    print(f"  Median:            {results['startup_ms_median']:.1f} ms")
    print(f"  Std Dev:           {results['startup_ms_stdev']:.1f} ms")
    print(f"  Min:               {results['startup_ms_min']:.1f} ms")
    print(f"  Max:               {results['startup_ms_max']:.1f} ms")
    print(f"\nMemory Usage:")
    print(f"  Python Heap:       {results['memory_python_mb_avg']:.2f} MB")
    print(f"  Process RSS:       {results['memory_rss_mb_avg']:.2f} MB")
    print(f"{'='*60}\n")


def compare_results(baseline: Dict, optimized: Dict):
    """Compare baseline vs optimized results"""
    startup_improvement = ((baseline['startup_ms_avg'] - optimized['startup_ms_avg']) 
                          / baseline['startup_ms_avg'] * 100)
    memory_change = ((optimized['memory_rss_mb_avg'] - baseline['memory_rss_mb_avg']) 
                     / baseline['memory_rss_mb_avg'] * 100)
    
    print(f"\n{'='*60}")
    print("COMPARISON: Baseline vs Optimized")
    print(f"{'='*60}")
    print(f"Startup Time:")
    print(f"  Baseline:          {baseline['startup_ms_avg']:.1f} ms")
    print(f"  Optimized:         {optimized['startup_ms_avg']:.1f} ms")
    print(f"  Improvement:       {startup_improvement:+.1f}% ({'✓' if startup_improvement > 0 else '✗'})")
    print(f"\nMemory Usage:")
    print(f"  Baseline:          {baseline['memory_rss_mb_avg']:.2f} MB")
    print(f"  Optimized:         {optimized['memory_rss_mb_avg']:.2f} MB")
    print(f"  Change:            {memory_change:+.1f}% ({'✓' if abs(memory_change) < 5 else '⚠'})")
    
    # Success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA:")
    print(f"{'='*60}")
    criteria = [
        ("Startup reduction > 50%", startup_improvement > 50, startup_improvement),
        ("Memory increase < 5%", abs(memory_change) < 5, memory_change)
    ]
    
    for criterion, passed, value in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {criterion:30} {status}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Profile Media Downloader startup performance')
    parser.add_argument('--mode', choices=['baseline', 'optimized', 'compare'], 
                       default='baseline', help='Profiling mode')
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Number of iterations to run')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # Load both baseline and optimized results
        baseline_file = Path('profile_baseline.json')
        optimized_file = Path('profile_optimized.json')
        
        if not baseline_file.exists() or not optimized_file.exists():
            print("Error: Missing profile files. Run with --mode baseline and --mode optimized first.")
            sys.exit(1)
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        with open(optimized_file) as f:
            optimized = json.load(f)
        
        print_results(baseline, "Baseline Results")
        print_results(optimized, "Optimized Results")
        compare_results(baseline, optimized)
    else:
        # Run profiling
        results = profile_startup(args.iterations)
        print_results(results, f"{args.mode.capitalize()} Results")
        
        # Save to file if requested
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = Path(f'profile_{args.mode}.json')
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
