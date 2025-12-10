from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageEnhance
import os
import time

def remove_bg(src, model_name='isnet-general-use', alpha_matting=False, post_process=False):
    try:
        img = Image.open(src)
        
        # Pre-processing: Resize if too small (upscale for better detection)
        if img.width < 1000:
             scale = 1000 / img.width
             new_size = (int(img.width * scale), int(img.height * scale))
             img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Pre-processing: Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)

        # Session
        session = new_session(model_name)
        
        # Remove
        if alpha_matting:
            output = remove(img, session=session, alpha_matting=True,
                            alpha_matting_foreground_threshold=240,
                            alpha_matting_background_threshold=10, 
                            alpha_matting_erode_size=10)
        else:
            output = remove(img, session=session)
            
        # Post-Processing
        if post_process:
             # Extract alpha
             alpha = output.split()[-1]
             # Light blur on alpha to smooth jagged edges
             alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1))
             # Contrast alpha to sharpen the soft edges
             enhancer = ImageEnhance.Contrast(alpha)
             alpha = enhancer.enhance(1.5)
             output.putalpha(alpha)

        # Save to temp
        temp_dir = os.path.join(os.getcwd(), 'screenshots')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        
        fname = f"bg_removed_{int(time.time()*1000)}.png"
        out_path = os.path.join(temp_dir, fname)
        output.save(out_path)
        
        return {'success': True, 'path': out_path}
    except Exception as e:
        return {'success': False, 'error': str(e)}
