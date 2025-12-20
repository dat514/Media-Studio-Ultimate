"""
Multi-Mode Background Remover Verification
Tests all three modes: remove_bg, blur_background, change_background
"""
import os
from api.bg_remover import remove_bg
from PIL import Image, ImageDraw

def create_test_images():
    """Create test images if they don't exist."""
    test_dir = os.path.join(os.getcwd(), 'test_images')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Create sample foreground image (person-like)
    fg_test = os.path.join(test_dir, 'test_subject.png')
    if not os.path.exists(fg_test):
        img = Image.new('RGB', (800, 600), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Draw a simple "person" shape
        draw.ellipse([300, 100, 500, 300], fill=(255, 200, 180))  # Head
        draw.rectangle([350, 280, 450, 500], fill=(100, 150, 200))  # Body
        img.save(fg_test)
        print(f"✅ Created: {fg_test}")
    
    # Create sample background image
    bg_test = os.path.join(test_dir, 'test_background.png')
    if not os.path.exists(bg_test):
        img = Image.new('RGB', (800, 600))
        draw = ImageDraw.Draw(img)
        # Gradient background
        for y in range(600):
            color = int(255 * (y / 600))
            draw.line([(0, y), (800, y)], fill=(color, 100, 255 - color))
        img.save(bg_test)
        print(f"✅ Created: {bg_test}")
    
    return fg_test, bg_test

def test_mode(mode_name, **kwargs):
    """Test a specific mode."""
    print(f"\n{'='*60}")
    print(f"Testing Mode: {mode_name.upper()}")
    print(f"{'='*60}")
    
    result = remove_bg(**kwargs)
    
    if result['success']:
        print(f"✅ SUCCESS!")
        print(f"   Output: {result['path']}")
        print(f"   Mode: {result.get('mode', 'N/A')}")
        
        if os.path.exists(result['path']):
            output_img = Image.open(result['path'])
            print(f"   Size: {output_img.size}")
            print(f"   Mode: {output_img.mode}")
        else:
            print("   ⚠️  Output file not found!")
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    return result['success']

def main():
    print("="*60)
    print("Multi-Mode Background Remover - Verification")
    print("="*60)
    
    # Create test images
    fg_image, bg_image = create_test_images()
    
    # Test Mode 1: Remove Background (Transparent)
    success1 = test_mode(
        "Remove Background",
        src=fg_image,
        mode='remove_bg'
    )
    
    # Test Mode 2: Blur Background
    success2 = test_mode(
        "Blur Background (radius=20)",
        src=fg_image,
        mode='blur_background',
        blur_radius=20
    )
    
    # Test Mode 3: Change Background
    success3 = test_mode(
        "Change Background",
        src=fg_image,
        mode='change_background',
        new_bg_path=bg_image
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Remove Background: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Blur Background: {'PASS' if success2 else 'FAIL'}")
    print(f"✅ Change Background: {'PASS' if success3 else 'FAIL'}")
    print("="*60)
    
    if success1 and success2 and success3:
        print("\n🎉 ALL TESTS PASSED! World Class Multi-Mode Ready!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")

if __name__ == '__main__':
    main()
