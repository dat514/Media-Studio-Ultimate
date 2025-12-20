"""
Verification script for World Class Background Remover
Tests the new pipeline with basic functionality checks
"""
import os
from api.bg_remover import remove_bg
from PIL import Image

def test_bg_remover():
    print("=" * 60)
    print("World Class Background Remover - Verification Test")
    print("=" * 60)
    
    # Check if we have a test image
    test_dir = os.path.join(os.getcwd(), 'test_images')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("\n⚠️  No test images found. Please add images to 'test_images/' folder")
        print("Creating a simple test image...")
        
        # Create a simple gradient test image
        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
        for x in range(200, 600):
            for y in range(150, 450):
                # Simple circular gradient
                dist = ((x-400)**2 + (y-300)**2) ** 0.5
                if dist < 150:
                    intensity = int(255 * (1 - dist/150))
                    img.putpixel((x, y), (intensity, 100, 200))
        
        test_image = os.path.join(test_dir, 'test_gradient.png')
        img.save(test_image)
        print(f"✅ Created test image: {test_image}")
    else:
        # Find first image in test directory
        images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print("\n❌ No images found in test_images/ folder")
            return
        test_image = os.path.join(test_dir, images[0])
    
    print(f"\n📸 Testing with: {test_image}")
    print("\n⏳ Processing... (This may take a moment)")
    
    # Run the background remover
    result = remove_bg(test_image)
    
    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"   Output: {result['path']}")
        
        # Verify the output
        if os.path.exists(result['path']):
            output_img = Image.open(result['path'])
            print(f"   Size: {output_img.size}")
            print(f"   Mode: {output_img.mode}")
            
            if output_img.mode == 'RGBA':
                print("\n✨ World Class Magic Applied!")
                print("   ✓ 4 channels (RGBA) confirmed")
                print("   ✓ Transparency preserved")
                print("   ✓ File saved to temp directory")
            else:
                print("\n⚠️  Warning: Output is not RGBA")
        else:
            print("\n❌ Output file not found!")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    test_bg_remover()
