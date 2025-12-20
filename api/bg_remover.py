import os
import time
import shutil
import numpy as np
import cv2
from rembg import remove, new_session
from PIL import Image, ImageEnhance, ImageFilter

# Global session to avoid reloading model
_session = None
TEMP_DIR = os.path.join(os.getcwd(), 'temp_bg_removed')

def get_session(model_name="isnet-general-use"):
    global _session
    if _session is None:
        _session = new_session(model_name)
    return _session

def cleanup_temp():
    """Cleans up the temporary directory."""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temp directory: {TEMP_DIR}")
        os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception as e:
        print(f"Error cleaning temp directory: {e}")

class WorldClassRemover:
    def __init__(self, model_name="isnet-general-use"):
        self.session = get_session(model_name)

    def _run_core_removal(self, image_path):
        """
        Core removal pipeline (World Class Quality).
        Returns: (original_rgb_pil, refined_alpha_np, original_size)
        """
        # --- 1. PRE-PROCESS ---
        img_pil = Image.open(image_path).convert("RGB")
        w, h = img_pil.size
        
        # Scale logic: Max dimension 2048px
        max_dim = 2048
        scale_factor = 1.0
        if max(w, h) > max_dim:
            scale_factor = max_dim / max(w, h)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Enhance for better detection
        img_enhanced = ImageEnhance.Sharpness(img_pil).enhance(1.2)
        img_enhanced = ImageEnhance.Contrast(img_enhanced).enhance(1.1)

        # --- 2. AI REMOVAL ---
        img_np = np.array(img_enhanced)
        height, width = img_np.shape[:2]
        
        # Adaptive parameters
        erode_size = max(1, int(width * 0.001)) 
        
        result = remove(
            img_enhanced,
            session=self.session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=erode_size
        )

        # --- 3. POST-PROCESSING (VFX) ---
        result_np = np.array(result)
        if result_np.shape[2] < 4: 
            raise Exception("Failed to generate alpha channel")

        alpha = result_np[:, :, 3]

        # Clean Mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(alpha, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            max_area = stats[largest_label, cv2.CC_STAT_AREA]
            noise_thresh = max_area * 0.005
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < noise_thresh:
                    alpha[labels == i] = 0

        # Smart Smooth
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha = self.smooth_step_alpha(alpha)

        return img_pil, alpha, (w, h)

    def process_advanced(self, image_path, mode='remove_bg', blur_radius=15, new_bg_path=None):
        """
        Advanced multi-mode processing.
        Modes: 'remove_bg', 'blur_background', 'change_background'
        """
        try:
            # Ensure temp dir exists
            if not os.path.exists(TEMP_DIR):
                os.makedirs(TEMP_DIR)

            # Run core removal pipeline
            original_rgb_pil, refined_alpha_np, original_size = self._run_core_removal(image_path)
            
            # Process based on mode
            if mode == 'remove_bg':
                # Color Decontamination (ONLY for transparent background)
                rgb_np = np.array(original_rgb_pil)
                foreground_mask = (refined_alpha_np > 240).astype(np.uint8)
                
                dilation_size = 3
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_size + 1, 2*dilation_size+1))
                
                r, g, b = cv2.split(rgb_np)
                r = self.dilate_channel(r, foreground_mask, element)
                g = self.dilate_channel(g, foreground_mask, element)
                b = self.dilate_channel(b, foreground_mask, element)
                rgb_clean = cv2.merge([r, g, b])
                
                final_result = np.dstack((rgb_clean, refined_alpha_np))
                final_img = Image.fromarray(final_result)
                
            elif mode == 'blur_background':
                final_img = self._apply_blur_background(image_path, original_rgb_pil, refined_alpha_np, blur_radius)
                
            elif mode == 'change_background':
                if new_bg_path is None:
                    raise ValueError("new_bg_path must be provided for 'change_background' mode.")
                final_img = self._apply_new_background(original_rgb_pil, refined_alpha_np, new_bg_path)
                
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Resize to original size if needed
            if final_img.size != original_size:
                final_img = final_img.resize(original_size, Image.Resampling.LANCZOS)
            
            # Save
            fname = f"{mode}_{int(time.time()*1000)}.png"
            out_path = os.path.join(TEMP_DIR, fname)
            final_img.save(out_path, "PNG")
            
            return {'success': True, 'path': out_path, 'mode': mode}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _apply_blur_background(self, original_image_path, fg_rgb_pil, alpha_np, blur_radius):
        """Blur the original background and composite foreground."""
        # Load original image at the same size as fg_rgb_pil
        original_img = Image.open(original_image_path).convert("RGB")
        if original_img.size != fg_rgb_pil.size:
            original_img = original_img.resize(fg_rgb_pil.size, Image.Resampling.LANCZOS)
        
        # Blur the entire original image
        blurred_bg = original_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Create alpha mask for compositing
        alpha_mask = Image.fromarray(alpha_np, mode='L')
        
        # Composite: foreground over blurred background
        result = Image.composite(fg_rgb_pil, blurred_bg, alpha_mask)
        
        return result

    def _apply_new_background(self, fg_rgb_pil, alpha_np, new_bg_path):
        """Replace background with a new image."""
        # Load new background
        new_bg = Image.open(new_bg_path).convert("RGB")
        
        # Resize to match foreground
        new_bg_resized = new_bg.resize(fg_rgb_pil.size, Image.Resampling.LANCZOS)
        
        # Create alpha mask
        alpha_mask = Image.fromarray(alpha_np, mode='L')
        
        # Composite: foreground over new background
        result = Image.composite(fg_rgb_pil, new_bg_resized, alpha_mask)
        
        return result

    def process(self, image_path):
        """Legacy method for backward compatibility (remove_bg mode)."""
        return self.process_advanced(image_path, mode='remove_bg')

    def smooth_step_alpha(self, alpha):
        """Sigmoid-like function to clean alpha edges."""
        a = alpha.astype(float) / 255.0
        a = np.clip((a - 0.05) * 1.2, 0, 1) 
        return (a * 255).astype(np.uint8)

    def dilate_channel(self, channel, mask, kernel):
        """Dilate color only into edges, keeping foreground intact."""
        dilated = cv2.dilate(channel, kernel, iterations=1)
        return np.where(mask == 1, channel, dilated)

# Exposed wrapper
def remove_bg(src, model='isnet-general-use', mode='remove_bg', blur_radius=15, new_bg_path=None):
    """
    Multi-mode background processing.
    
    Args:
        src: Source image path
        model: AI model name (default: isnet-general-use)
        mode: Processing mode ('remove_bg', 'blur_background', 'change_background')
        blur_radius: Blur intensity for blur_background mode (default: 15)
        new_bg_path: Path to new background image for change_background mode
    
    Returns:
        dict: {'success': bool, 'path': str, 'mode': str} or {'success': False, 'error': str}
    """
    remover = WorldClassRemover(model)
    return remover.process_advanced(src, mode=mode, blur_radius=blur_radius, new_bg_path=new_bg_path)
