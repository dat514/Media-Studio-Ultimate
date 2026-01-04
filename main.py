import webview
import threading
import time
import sys
import os

USE_LAZY_IMPORTS = os.getenv('LAZY_IMPORTS', 'true').lower() == 'true'

if USE_LAZY_IMPORTS:
    from lazy_import import lazy_import
    from api import base
else:
    from api import base, downloader, converter, editor, gif, shortener, bg_remover, wave_auth

from server import run_server
import ui

class Api:
    def __init__(self):
        self._modules_loaded = set()
    
    def _get_downloader(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('downloader', lambda: __import__('api.downloader', fromlist=['downloader']), 'Downloader (yt-dlp)')
        else:
            return downloader
    
    def _get_converter(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('converter', lambda: __import__('api.converter', fromlist=['converter']), 'Converter (FFmpeg)')
        else:
            return converter
    
    def _get_editor(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('editor', lambda: __import__('api.editor', fromlist=['editor']), 'Editor')
        else:
            return editor
    
    def _get_gif(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('gif', lambda: __import__('api.gif', fromlist=['gif']), 'GIF Maker')
        else:
            return gif
    
    def _get_shortener(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('shortener', lambda: __import__('api.shortener', fromlist=['shortener']), 'URL Shortener & QR Generator')
        else:
            return shortener
    
    def _get_bg_remover(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('bg_remover', lambda: __import__('api.bg_remover', fromlist=['bg_remover']), 'Background Remover (Rembg)')
        else:
            return bg_remover
    
    def _get_wave_auth(self):
        if USE_LAZY_IMPORTS:
            return lazy_import('wave_auth', lambda: __import__('api.wave_auth', fromlist=['wave_auth']), 'Wave Auth (Librosa)')
        else:
            return wave_auth
    
    def update_libraries(self):
        return base.update_libraries()
    
    def choose_files(self, multi):
        return base.choose_files(multi)
    
    def choose_folder(self):
        return base.choose_folder()
    
    def open_directory(self, path):
        base.open_directory(path)
    
    def open_folder(self, path):
        base.open_directory(path)
    
    def file_exists(self, path):
        return os.path.exists(path)
    
    def analyze(self, url):
        return self._get_downloader().analyze(url)
    
    def download(self, url, opts):
        return self._get_downloader().download(url, opts)
    
    def convert(self, src, fmt, folder):
        return self._get_converter().convert(src, fmt, folder)
    
    def edit_media(self, args):
        return self._get_editor().edit_media(args)
    
    def make_gif(self, args):
        return self._get_gif().make_gif(args)
    
    def shorten_url(self, url, alias=None):
        return self._get_shortener().shorten_url(url, alias)
    
    def generate_advanced_qr(self, args):
        return self._get_shortener().generate_advanced_qr(args)
    
    def save_qr_cleanup(self, src, folder):
        return self._get_shortener().save_qr_cleanup(src, folder)
    
    def remove_bg(self, src, model='isnet-general-use', mode='remove_bg', blur_radius=15, new_bg_path=None):
        return self._get_bg_remover().remove_bg(src, model, mode, blur_radius, new_bg_path)
    
    def bg_edit(self, session_id, strokes, display_size):
        return self._get_bg_remover().edit_mask(session_id, strokes, display_size)
    
    def bg_undo(self, session_id):
        return self._get_bg_remover().undo(session_id)
    
    def bg_redo(self, session_id):
        return self._get_bg_remover().redo(session_id)
    
    def bg_save_export(self, session_id, folder):
        return self._get_bg_remover().save_result(session_id, folder)
    
    def wa_analyze(self, path):
        return self._get_wave_auth().analyze_audio(path)
    
    def wa_history(self):
        return self._get_wave_auth().get_history()
    
    def wa_clear(self):
        return self._get_wave_auth().clear_history()
    
    def wa_export_csv(self):
        return self._get_wave_auth().get_history_csv()
    
    def wa_clear_temp(self):
        return self._get_wave_auth().clear_temp_images()

def main():
    if not USE_LAZY_IMPORTS:
        from api.bg_remover import cleanup_temp
        cleanup_temp()
    
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1)
    webview.create_window("Media Studio Ultimate 2.4", "http://127.0.0.1:8000", width=1150, height=850, background_color='#0a0e17', js_api=Api())
    webview.start(debug=False)

if __name__ == '__main__':
    main()
