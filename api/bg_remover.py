import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
from datetime import datetime, timedelta
import re
import json
import logging
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import base64
from io import BytesIO
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    mask_base64: str
    timestamp: datetime
    action: str


class SecurityValidator:
    VALID_SESSION_ID_PATTERN = r'^[a-zA-Z0-9_-]{8,64}$'
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    @classmethod
    def validate_session_id(cls, session_id: str) -> str:
        if not re.match(cls.VALID_SESSION_ID_PATTERN, session_id):
            logger.error(f"Invalid session ID: {session_id}")
            raise ValueError("Invalid session ID format")
        return session_id
    
    @classmethod
    def validate_path(cls, path: str, base_dir: str) -> str:
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base_dir)
        
        if not abs_path.startswith(abs_base):
            raise ValueError("Path traversal detected")
        
        return abs_path
    
    @classmethod
    def validate_file_size(cls, file_path: str) -> bool:
        size = os.path.getsize(file_path)
        if size > cls.MAX_FILE_SIZE:
            raise ValueError(f"File size {size} exceeds limit")
        return True


class CoordinateScaler:
    def __init__(self):
        self._scale_cache: Dict[str, Tuple[float, float]] = {}
    
    def scale_points(
        self,
        points: List[Tuple[int, int]],
        from_size: Tuple[int, int],
        to_size: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        if from_size == to_size:
            return points
        
        cache_key = f"{from_size}_{to_size}"
        
        if cache_key not in self._scale_cache:
            scale_x = to_size[0] / from_size[0]
            scale_y = to_size[1] / from_size[1]
            self._scale_cache[cache_key] = (scale_x, scale_y)
        
        scale_x, scale_y = self._scale_cache[cache_key]
        
        return [
            (int(x * scale_x), int(y * scale_y))
            for x, y in points
        ]
    
    def clear_cache(self):
        self._scale_cache.clear()


class SessionHistory:
    def __init__(self, max_size: int = 20):
        self.undo_stack: deque = deque(maxlen=max_size)
        self.redo_stack: deque = deque(maxlen=max_size)
    
    def commit(self, state: SessionState):
        self.undo_stack.append(state)
        self.redo_stack.clear()
    
    def undo(self) -> Optional[SessionState]:
        if len(self.undo_stack) < 2:
            return None
        
        current = self.undo_stack.pop()
        self.redo_stack.append(current)
        return self.undo_stack[-1]
    
    def redo(self) -> Optional[SessionState]:
        if not self.redo_stack:
            return None
        
        state = self.redo_stack.pop()
        self.undo_stack.append(state)
        return state
    
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 1
    
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0
    
    def clear(self):
        self.undo_stack.clear()
        self.redo_stack.clear()


class ResourceManager:
    def __init__(self, temp_dir: str, ttl_hours: int = 24):
        self.temp_dir = temp_dir
        self.ttl_hours = ttl_hours
        self._ensure_directory()
    
    def _ensure_directory(self):
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def get_session_path(self, session_id: str, suffix: str) -> str:
        session_id = SecurityValidator.validate_session_id(session_id)
        filename = f"{session_id}_{suffix}"
        path = os.path.join(self.temp_dir, filename)
        return SecurityValidator.validate_path(path, self.temp_dir)
    
    def cleanup_old_files(self) -> int:
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        removed_count = 0
        
        try:
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        logger.info(f"Removed old file: {filename}")
                    except OSError as e:
                        logger.error(f"Failed to remove {filename}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleanup: removed {removed_count} files")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        return removed_count
    
    def cleanup_session(self, session_id: str) -> int:
        session_id = SecurityValidator.validate_session_id(session_id)
        removed_count = 0
        
        try:
            for filename in os.listdir(self.temp_dir):
                if filename.startswith(session_id):
                    file_path = os.path.join(self.temp_dir, filename)
                    os.remove(file_path)
                    removed_count += 1
            
            logger.info(f"Session {session_id} cleanup: {removed_count} files")
        
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            raise
        
        return removed_count
    
    def session_exists(self, session_id: str) -> bool:
        session_id = SecurityValidator.validate_session_id(session_id)
        
        for filename in os.listdir(self.temp_dir):
            if filename.startswith(session_id):
                return True
        
        return False


class ImageProcessor:
    @staticmethod
    def mask_to_base64(mask: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', mask)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def base64_to_mask(mask_b64: str) -> np.ndarray:
        mask_data = base64.b64decode(mask_b64)
        return cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    
    @staticmethod
    def image_to_base64(image: np.ndarray, format: str = 'jpeg', quality: int = 85) -> str:
        if format.lower() == 'jpeg':
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            mime = 'image/jpeg'
        else:
            _, buffer = cv2.imencode('.png', image)
            mime = 'image/png'
        
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:{mime};base64,{img_b64}"
    
    @staticmethod
    def generate_preview_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        color = (0, 255, 0)
        
        mask_bool = mask > 128
        overlay[mask_bool] = cv2.addWeighted(
            image[mask_bool], 0.6,
            np.full_like(image[mask_bool], color), 0.4,
            0
        )
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        
        return overlay
    
    @staticmethod
    def apply_mask_with_alpha(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image)
        alpha = cv2.GaussianBlur(mask, (3, 3), 0)
        return cv2.merge([b, g, r, alpha])

    @staticmethod
    def blur_background(image: np.ndarray, mask: np.ndarray, blur_radius: int = 15) -> np.ndarray:
        ksize = (blur_radius * 2) + 1
        blurred_bg = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        mask_expanded = np.dstack([mask, mask, mask]) / 255.0
        
        result = (image * mask_expanded + blurred_bg * (1.0 - mask_expanded)).astype(np.uint8)
        return result

    @staticmethod
    def replace_background(image: np.ndarray, mask: np.ndarray, new_bg_path: str) -> np.ndarray:
        if not os.path.exists(new_bg_path):
            raise FileNotFoundError(f"Background image not found: {new_bg_path}")
            
        new_bg = cv2.imread(new_bg_path)
        if new_bg is None:
            raise ValueError("Failed to load background image")
            
        h, w = image.shape[:2]
        new_bg = cv2.resize(new_bg, (w, h))
        
        mask_expanded = np.dstack([mask, mask, mask]) / 255.0
        result = (image * mask_expanded + new_bg * (1.0 - mask_expanded)).astype(np.uint8)
        return result
    
    @staticmethod
    def apply_strokes_to_mask(
        mask: np.ndarray,
        strokes: List[Dict],
        coordinate_scaler: Optional[CoordinateScaler] = None,
        display_size: Optional[Tuple[int, int]] = None,
        original_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        mask_copy = mask.copy()
        
        for stroke in strokes:
            points = stroke['points']
            mode = stroke['mode']
            radius = stroke.get('radius', 20)
            
            if coordinate_scaler and display_size and original_size:
                points = coordinate_scaler.scale_points(points, display_size, original_size)
                scale_factor = original_size[0] / display_size[0]
                radius = int(radius * scale_factor)
            
            if mode == 'restore' or mode == 'add':
                color = 255
            else:  
                color = 0
            
            points_np = np.array(points, dtype=np.int32)
            
            if len(points_np) == 0:
                continue
                
            if len(points_np) == 1:
                cv2.circle(mask_copy, tuple(points_np[0]), radius, color, -1, cv2.LINE_AA)
            else:
                for i in range(len(points_np)):
                    cv2.circle(mask_copy, tuple(points_np[i]), radius, color, -1, cv2.LINE_AA)
                    
                    if i < len(points_np) - 1:
                        cv2.line(
                            mask_copy,
                            tuple(points_np[i]),
                            tuple(points_np[i + 1]),
                            color,
                            radius * 2,  
                            cv2.LINE_AA
                        )
        
        if mask_copy.max() > 0:  
            mask_copy = cv2.GaussianBlur(mask_copy, (3, 3), 0)
        
        return mask_copy


class BackgroundRemovalSystem:
    def __init__(self, temp_dir: str = "editor_temp", model_name: str = "u2net"):
        self.temp_dir = temp_dir
        self.resource_manager = ResourceManager(temp_dir)
        self.coordinate_scaler = CoordinateScaler()
        self.session_histories: Dict[str, SessionHistory] = {}
        self.session_metadata: Dict[str, Dict] = {}
        self.model_name = model_name
        
        try:
            self.rembg_session = new_session(model_name)
            logger.info(f"Initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize rembg: {e}")
            raise
    
    def _get_or_create_history(self, session_id: str) -> SessionHistory:
        if session_id not in self.session_histories:
            self.session_histories[session_id] = SessionHistory()
        return self.session_histories[session_id]
    
    def generate_initial_mask(
        self,
        image_path: str,
        session_id: str,
        return_format: str = 'base64'
    ) -> Dict[str, Any]:
        try:
            session_id = SecurityValidator.validate_session_id(session_id)
            SecurityValidator.validate_file_size(image_path)
            
            original_path = self.resource_manager.get_session_path(session_id, "original.jpg")
            img = Image.open(image_path).convert('RGB')
            img.save(original_path)
            
            self.session_metadata[session_id] = {
                'original_size': img.size,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Generating mask for session: {session_id}")
            output = remove(img, session=self.rembg_session, only_mask=True)
            
            mask_np = np.array(output)
            if len(mask_np.shape) == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
            
            mask_path = self.resource_manager.get_session_path(session_id, "mask.png")
            cv2.imwrite(mask_path, mask_np)
            
            history = self._get_or_create_history(session_id)
            state = SessionState(
                mask_base64=ImageProcessor.mask_to_base64(mask_np),
                timestamp=datetime.now(),
                action='initial'
            )
            history.commit(state)
            
            preview_url = self._generate_preview_response(
                original_path,
                mask_np,
                return_format
            )
            
            logger.info(f"Mask generation complete: {session_id}")
            
            return {
                'success': True,
                'preview': preview_url,
                'session_id': session_id,
                'can_undo': history.can_undo(),
                'can_redo': history.can_redo()
            }
        
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_preview_response(
        self,
        image_path: str,
        mask: np.ndarray,
        return_format: str
    ) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        overlay = ImageProcessor.generate_preview_overlay(img, mask)
        
        if return_format == 'base64':
            return ImageProcessor.image_to_base64(overlay)
        elif return_format == 'file_path':
            return image_path
        else:
            return ImageProcessor.image_to_base64(overlay)
    
    def edit_mask(
        self,
        session_id: str,
        strokes: List[Dict],
        display_size: Optional[Tuple[int, int]] = None,
        return_format: str = 'base64'
    ) -> Dict[str, Any]:
        try:
            session_id = SecurityValidator.validate_session_id(session_id)
            
            mask_path = self.resource_manager.get_session_path(session_id, "mask.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise FileNotFoundError(f"Mask not found for session: {session_id}")
            
            original_size = self.session_metadata[session_id]['original_size']
            
            updated_mask = ImageProcessor.apply_strokes_to_mask(
                mask,
                strokes,
                self.coordinate_scaler,
                display_size,
                original_size
            )
            
            cv2.imwrite(mask_path, updated_mask)
            
            history = self._get_or_create_history(session_id)
            state = SessionState(
                mask_base64=ImageProcessor.mask_to_base64(updated_mask),
                timestamp=datetime.now(),
                action=strokes[0]['mode'] if strokes else 'edit'
            )
            history.commit(state)
            
            original_path = self.resource_manager.get_session_path(session_id, "original.jpg")
            preview = self._generate_preview_response(
                original_path,
                updated_mask,
                return_format
            )
            
            logger.info(f"Mask edited: {session_id}, strokes: {len(strokes)}")
            
            return {
                'success': True,
                'preview': preview,
                'can_undo': history.can_undo(),
                'can_redo': history.can_redo()
            }
        
        except Exception as e:
            logger.error(f"Mask edit failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def undo(self, session_id: str, return_format: str = 'base64') -> Dict[str, Any]:
        try:
            session_id = SecurityValidator.validate_session_id(session_id)
            history = self._get_or_create_history(session_id)
            
            previous_state = history.undo()
            
            if previous_state is None:
                return {
                    'success': False,
                    'error': 'No history to undo'
                }
            
            mask = ImageProcessor.base64_to_mask(previous_state.mask_base64)
            mask_path = self.resource_manager.get_session_path(session_id, "mask.png")
            cv2.imwrite(mask_path, mask)
            
            original_path = self.resource_manager.get_session_path(session_id, "original.jpg")
            preview = self._generate_preview_response(original_path, mask, return_format)
            
            logger.info(f"Undo successful: {session_id}")
            
            return {
                'success': True,
                'preview': preview,
                'can_undo': history.can_undo(),
                'can_redo': history.can_redo()
            }
        
        except Exception as e:
            logger.error(f"Undo failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def redo(self, session_id: str, return_format: str = 'base64') -> Dict[str, Any]:
        try:
            session_id = SecurityValidator.validate_session_id(session_id)
            history = self._get_or_create_history(session_id)
            
            next_state = history.redo()
            
            if next_state is None:
                return {
                    'success': False,
                    'error': 'No history to redo'
                }
            
            mask = ImageProcessor.base64_to_mask(next_state.mask_base64)
            mask_path = self.resource_manager.get_session_path(session_id, "mask.png")
            cv2.imwrite(mask_path, mask)
            
            original_path = self.resource_manager.get_session_path(session_id, "original.jpg")
            preview = self._generate_preview_response(original_path, mask, return_format)
            
            logger.info(f"Redo successful: {session_id}")
            
            return {
                'success': True,
                'preview': preview,
                'can_undo': history.can_undo(),
                'can_redo': history.can_redo()
            }
        
        except Exception as e:
            logger.error(f"Redo failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def remove_background(
        self,
        session_id: str,
        output_format: str = 'png',
        return_format: str = 'base64',
        mode: str = 'remove_bg',
        blur_radius: int = 15,
        new_bg_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            session_id = SecurityValidator.validate_session_id(session_id)
            
            original_path = self.resource_manager.get_session_path(session_id, "original.jpg")
            mask_path = self.resource_manager.get_session_path(session_id, "mask.png")
            
            img = cv2.imread(original_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                raise FileNotFoundError(f"Files not found for session: {session_id}")
            
            if mode == 'remove_bg':
                result = ImageProcessor.apply_mask_with_alpha(img, mask)
                result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
            
            elif mode == 'blur_background':
                result = ImageProcessor.blur_background(img, mask, blur_radius)
                result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                result_pil.putalpha(255)
            
            elif mode == 'change_background':
                if not new_bg_path:
                    raise ValueError("new_bg_path required for change_background mode")
                result = ImageProcessor.replace_background(img, mask, new_bg_path)
                result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                result_pil.putalpha(255)
            
            else:
                raise ValueError(f"Unknown mode: {mode}")

            
            if return_format == 'base64':
                buffer = BytesIO()
                
                if output_format.lower() == 'jpg' and mode == 'remove_bg':
                    white_bg = Image.new('RGB', result_pil.size, (255, 255, 255))
                    white_bg.paste(result_pil, mask=result_pil.split()[3])
                    white_bg.save(buffer, format='JPEG', quality=95)
                    mime_type = 'image/jpeg'
                elif output_format.lower() == 'jpg':
                    result_rgb = result_pil.convert('RGB')
                    result_rgb.save(buffer, format='JPEG', quality=95)
                    mime_type = 'image/jpeg'
                else:
                    result_pil.save(buffer, format='PNG')
                    mime_type = 'image/png'
                
                img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                result_data = f"data:{mime_type};base64,{img_b64}"
            
            else:
                output_path = self.resource_manager.get_session_path(
                    session_id,
                    f"result_{int(datetime.now().timestamp())}.{output_format}"
                )
                
                if output_format.lower() == 'jpg':
                     if result_pil.mode == 'RGBA':
                        white_bg = Image.new('RGB', result_pil.size, (255, 255, 255))
                        white_bg.paste(result_pil, mask=result_pil.split()[3])
                        result_to_save = white_bg
                     else:
                        result_to_save = result_pil.convert('RGB')
                     
                     result_to_save.save(output_path, format='JPEG', quality=95)
                else:
                    result_pil.save(output_path, format='PNG')
                
                result_data = output_path
            
            logger.info(f"Background removed: {session_id}")
            
            return {
                'success': True,
                'result': result_data,
                'format': output_format,
                'mode': mode
            }
        
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup_old_files(self) -> int:
        return self.resource_manager.cleanup_old_files()
    
    def cleanup_session(self, session_id: str) -> int:
        session_id = SecurityValidator.validate_session_id(session_id)
        
        if session_id in self.session_histories:
            self.session_histories[session_id].clear()
            del self.session_histories[session_id]
        
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        
        return self.resource_manager.cleanup_session(session_id)
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        session_id = SecurityValidator.validate_session_id(session_id)
        
        exists = self.resource_manager.session_exists(session_id)
        
        if not exists:
            return {
                'exists': False
            }
        
        history = self._get_or_create_history(session_id)
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            'exists': True,
            'can_undo': history.can_undo(),
            'can_redo': history.can_redo(),
            'history_size': len(history.undo_stack),
            'metadata': metadata
        }


_shared_system: Optional[BackgroundRemovalSystem] = None

def _get_system(model_name: str = "u2net") -> BackgroundRemovalSystem:
    global _shared_system
    if _shared_system is None or _shared_system.model_name != model_name:
        _shared_system = BackgroundRemovalSystem(model_name=model_name, temp_dir="temp_bg_removed")
    return _shared_system

def cleanup_temp():
    try:
        sys = _get_system()
        sys.cleanup_old_files()
        logger.info("Executed legacy cleanup_temp")
    except Exception as e:
        logger.error(f"Legacy cleanup failed: {e}")

def remove_bg(src, model='isnet-general-use', mode='remove_bg', blur_radius=15, new_bg_path=None):
    import uuid
    
    sys = _get_system(model_name=model)
    session_id = f"legacy_{uuid.uuid4().hex}"[:20] 
    
    try:
        init_res = sys.generate_initial_mask(src, session_id, return_format='base64')
        if not init_res['success']:
            return init_res
            
        final_res = sys.remove_background(
            session_id=session_id,
            output_format='png',
            return_format='file_path',
            mode=mode,
            blur_radius=blur_radius,
            new_bg_path=new_bg_path
        )
        
        if final_res['success']:
            return {
                'success': True,
                'path': final_res['result'],
                'mode': mode,
                'session_id': session_id
            }
        else:
            return final_res

    except Exception as e:
        return {'success': False, 'error': str(e)}

def edit_mask(session_id, strokes):
    try:
        sys = _get_system()
        return sys.edit_mask(session_id, strokes)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def undo(session_id):
    try:
        sys = _get_system()
        return sys.undo(session_id)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def redo(session_id):
    try:
        sys = _get_system()
        return sys.redo(session_id)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_result(session_id, folder_path):
    try:
        import shutil
        import os
        sys = _get_system()
        
        res = sys.remove_background(
            session_id=session_id,
            output_format='png',
            return_format='file_path' 
        )
        
        if not res['success']:
            return res
            
        src_path = res['result']
        filename = os.path.basename(src_path)
        dst_path = os.path.join(folder_path, filename)
        
        shutil.copy2(src_path, dst_path)
        return {'success': True, 'path': dst_path}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    system = BackgroundRemovalSystem()
    system.cleanup_old_files()
