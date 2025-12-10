import webview
import threading
import time
import sys
import os

# Import modules
from api import base, downloader, converter, editor, gif, shortener, bg_remover
from server import run_server

# Import UI (referenced by server, but good to ensure valid)
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
    
    def remove_bg(self, src, model='isnet-general-use', alpha=False): 
        return bg_remover.remove_bg(src, model, alpha)

def main():
    # Start Server in background thread
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    
    # Give server a moment to start
    time.sleep(1) 
    
    # Create Window
    # Pointing to local server which serves ui.HTML_CONTENT
    webview.create_window(
        "Media Studio Ultimate 2.2", 
        "http://127.0.0.1:8000", 
        width=1150, height=850, 
        background_color='#0a0e17', 
        js_api=Api()
    )
    
    # Start App
    webview.start(debug=False)

if __name__ == '__main__':
    main()
