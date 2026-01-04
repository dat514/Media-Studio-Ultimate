import os
import sys
import subprocess
import webview
from webview import FileDialog

def _ff():
    local = os.path.join(os.getcwd(), 'ffmpeg.exe')
    return local if os.path.exists(local) else 'ffmpeg'

def _open_folder(path):
    import platform
    if platform.system() == "Windows":
        os.startfile(os.path.dirname(path) if os.path.isfile(path) else path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "--", os.path.dirname(path) if os.path.isfile(path) else path])
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) if os.path.isfile(path) else path])

def choose_files(allow_multiple=False):
    res = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=allow_multiple)
    return res if res else None

def choose_folder():
    res = webview.windows[0].create_file_dialog(FileDialog.FOLDER)
    return res[0] if res else None

def update_libraries():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp", "instaloader", "librosa", "numpy", "matplotlib", "soundfile", "pandas"], check=True)
        return {'success': True, 'message': 'Libraries updated successfully!'}
    except Exception as e:
        return {'success': False, 'message': f'Update failed: {e}'}

def open_directory(path):
    _open_folder(path)
