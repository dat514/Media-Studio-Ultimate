import webview
import threading
import time
import sys
import os
from api import base, downloader, converter, editor, gif, shortener, bg_remover, wave_auth
from server import run_server
import ui

class Api:
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
        return downloader.analyze(url)
    def download(self, url, opts):
        return downloader.download(url, opts)
    def convert(self, src, fmt, folder):
        return converter.convert(src, fmt, folder)
    def edit_media(self, args):
        return editor.edit_media(args)
    def make_gif(self, args):
        return gif.make_gif(args)
    def shorten_url(self, url, alias=None):
        return shortener.shorten_url(url, alias)
    def generate_advanced_qr(self, args):
        return shortener.generate_advanced_qr(args)
    def save_qr_cleanup(self, src, folder):
        return shortener.save_qr_cleanup(src, folder)
    def remove_bg(self, src, model='isnet-general-use', mode='remove_bg', blur_radius=15, new_bg_path=None):
        return bg_remover.remove_bg(src, model, mode, blur_radius, new_bg_path)
    def bg_edit(self, session_id, strokes, display_size):
        return bg_remover.edit_mask(session_id, strokes, display_size)
    def bg_undo(self, session_id):
        return bg_remover.undo(session_id)
    def bg_redo(self, session_id):
        return bg_remover.redo(session_id)
    def bg_save_export(self, session_id, folder):
        return bg_remover.save_result(session_id, folder)
    def wa_analyze(self, path):
        return wave_auth.analyze_audio(path)
    def wa_history(self):
        return wave_auth.get_history()
    def wa_clear(self):
        return wave_auth.clear_history()
    def wa_export_csv(self):
        return wave_auth.get_history_csv()
    def wa_clear_temp(self):
        return wave_auth.clear_temp_images()

def main():
    from api.bg_remover import cleanup_temp
    cleanup_temp()
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(1)
    webview.create_window("Media Studio Ultimate 2.4", "http://127.0.0.1:8000", width=1150, height=850, background_color='#0a0e17', js_api=Api())
    webview.start(debug=False)

if __name__ == '__main__':
    main()
