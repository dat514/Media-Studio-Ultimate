import sys
import time

print("="*60)
print("LAZY IMPORT VERIFICATION TEST")
print("="*60)

print("\nTest 1: Checking initial imports...")
start = time.time()

modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith('api')]
for mod in modules_to_remove:
    del sys.modules[mod]

import main

elapsed = (time.time() - start) * 1000
print(f"✓ Main module imported in {elapsed:.1f}ms")

# Check which API modules are loaded
api_modules = [m for m in sys.modules.keys() if m.startswith('api.') and m != 'api.base']
print(f"\nHeavy API modules loaded at startup: {len(api_modules)}")
for mod in sorted(api_modules):
    print(f"  - {mod}")

if len(api_modules) == 0:
    print("✓ SUCCESS: No heavy modules loaded at startup!")
else:
    print(f"⚠ WARNING: {len(api_modules)} modules loaded (expected 0 with lazy imports)")

# Test 2: Lazy loading on demand
print("\n" + "="*60)
print("Test 2: Testing lazy loading on first use...")
print("="*60)

api = main.Api()

# Test downloader (yt-dlp)
print("\n[1] Testing downloader module...")
modules_before = len([m for m in sys.modules.keys() if 'yt_dlp' in m or 'api.downloader' in m])
start = time.time()
try:
    api.analyze("https://www.youtube.com/watch?v=test")
except:
    pass  # Expected to fail, just testing import
elapsed = (time.time() - start) * 1000
modules_after = len([m for m in sys.modules.keys() if 'yt_dlp' in m or 'api.downloader' in m])

if modules_after > modules_before:
    print(f"✓ Module loaded on demand (took {elapsed:.1f}ms)")
else:
    print(f"✗ Module was already loaded")

# Test Background Remover (rembg, cv2)
print("\n[2] Testing bg_remover module...")
modules_before = len([m for m in sys.modules.keys() if any(x in m for x in ['cv2', 'rembg', 'bg_remover'])])
start = time.time()
try:
    api.remove_bg("test.png")
except:
    pass
elapsed = (time.time() - start) * 1000
modules_after = len([m for m in sys.modules.keys() if any(x in m for x in ['cv2', 'rembg', 'bg_remover'])])

if modules_after > modules_before:
    print(f"✓ Module loaded on demand (took {elapsed:.1f}ms)")
else:
    print(f"✗ Module was already loaded")

# Test Wave Auth (librosa)
print("\n[3] Testing wave_auth module...")
modules_before = len([m for m in sys.modules.keys() if any(x in m for x in ['librosa', 'wave_auth'])])
start = time.time()
try:
    api.wa_history()
except:
    pass
elapsed = (time.time() - start) * 1000
modules_after = len([m for m in sys.modules.keys() if any(x in m for x in ['librosa', 'wave_auth'])])

if modules_after > modules_before:
    print(f"✓ Module loaded on demand (took {elapsed:.1f}ms)")
else:
    print(f"✗ Module was already loaded")

print("\n" + "="*60)
print("Test completed!")
print("="*60)
