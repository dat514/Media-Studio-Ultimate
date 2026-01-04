import sys
import threading
import logging
from typing import Callable, Any, Optional, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LazyImporter:
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._main_lock = threading.Lock()
    
    def import_module(
        self, 
        module_name: str, 
        import_func: Callable, 
        error_context: Optional[str] = None
    ) -> Any:
        if module_name in self._cache:
            logger.debug(f"Using cached module: {module_name}")
            return self._cache[module_name]
        
        with self._main_lock:
            if module_name not in self._locks:
                self._locks[module_name] = threading.Lock()
        
        with self._locks[module_name]:
            if module_name in self._cache:
                logger.debug(f"Using cached module (post-lock): {module_name}")
                return self._cache[module_name]
            
            try:
                logger.info(f"Lazy loading module: {module_name}")
                start_time = __import__('time').time()
                
                result = import_func()
                
                elapsed_ms = (__import__('time').time() - start_time) * 1000
                logger.info(f"Successfully loaded {module_name} in {elapsed_ms:.1f}ms")
                
                self._cache[module_name] = result
                return result
                
            except ImportError as e:
                context = error_context or module_name
                logger.error(f"Failed to import {module_name}: {e}", exc_info=True)
                raise ImportError(
                    f"Failed to load {context}. "
                    f"Please check dependencies: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error importing {module_name}: {e}", 
                    exc_info=True
                )
                raise
    
    def clear_cache(self, module_name: Optional[str] = None) -> None:
        with self._main_lock:
            if module_name:
                removed = self._cache.pop(module_name, None)
                if removed:
                    logger.info(f"Cleared cache for: {module_name}")
            else:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all cached modules ({count} items)")
    
    def get_cache_status(self) -> Dict[str, bool]:
        return {name: True for name in self._cache.keys()}


_lazy_importer = LazyImporter()


def lazy_import(
    module_name: str, 
    import_func: Callable, 
    error_context: Optional[str] = None
) -> Any:
    return _lazy_importer.import_module(module_name, import_func, error_context)


def clear_cache(module_name: Optional[str] = None) -> None:
    _lazy_importer.clear_cache(module_name)


def get_cache_status() -> Dict[str, bool]:
    return _lazy_importer.get_cache_status()
