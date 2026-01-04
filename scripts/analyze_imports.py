import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def analyze(filename: str) -> None:
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    
    imports = defaultdict(lambda: {'self': 0, 'cumulative': 0})
    pattern = r'import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.+)'
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            self_time = int(match.group(1))
            cumul_time = int(match.group(2))
            module = match.group(3).strip()
            imports[module] = {
                'self': self_time / 1000000,  # Convert to ms
                'cumulative': cumul_time / 1000000
            }
    
    if not imports:
        print("Warning: No import time data found in file")
        print("Make sure you ran: python -X importtime main.py 2> import_profile.txt")
        return
    
    # Sort by cumulative time
    sorted_imports = sorted(
        imports.items(),
        key=lambda x: x[1]['cumulative'],
        reverse=True
    )
    
    print("\n" + "="*75)
    print("IMPORT TIME ANALYSIS")
    print("="*75)
    print(f"{'Module':<50} | {'Self (ms)':>10} | {'Total (ms)':>10}")
    print("-" * 75)
    
    # Show top 20 slowest imports
    for module, times in sorted_imports[:20]:
        # Truncate long module names
        display_module = module if len(module) <= 50 else module[:47] + "..."
        print(f"{display_module:<50} | {times['self']:>10.1f} | {times['cumulative']:>10.1f}")
    
    total_time = max(m['cumulative'] for m in imports.values())
    print("="*75)
    print(f"Total import time: {total_time:.1f}ms")
    print("="*75)
    
    # Identify major bottlenecks (>100ms)
    bottlenecks = [(m, t) for m, t in sorted_imports if t['cumulative'] > 100]
    if bottlenecks:
        print("\n⚠ BOTTLENECKS (imports >100ms):")
        for module, times in bottlenecks:
            print(f"  • {module}: {times['cumulative']:.1f}ms")
    
    print()


def main():
    if len(sys.argv) > 1:
        analyze(sys.argv[1])
    else:
        print("Usage: python analyze_imports.py <import_profile.txt>")
        print("\nTo generate import profile:")
        print("  python -X importtime main.py 2> import_profile.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()
