import yt_dlp
import json
import os
from .base import choose_folder, _open_folder

def analyze(url):
    with yt_dlp.YoutubeDL({'quiet':True}) as ydl:
        return json.dumps(ydl.extract_info(url, download=False), ensure_ascii=False)

def download(url, opts_json):
    try:
        folder = choose_folder()
        if not folder: return {'success': False}
        opts_data = json.loads(opts_json)
        dtype = opts_data['type']
        is_playlist = opts_data.get('is_playlist', False)
        
        ydl_opts = {
            'outtmpl': os.path.join(folder, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        if dtype == 'video':
            res = opts_data['res']
            fps = opts_data['fps']
            
            # Construct format string
            fmt_str = "bestvideo"
            filters = []
            
            if res != 'best':
                filters.append(f"height<={res}")
            if fps != 'best':
                filters.append(f"fps<={fps}")
            
            if filters:
                fmt_str += f"[{']['.join(filters)}]"
            
            fmt_str += "+bestaudio/best"
            
            ydl_opts['format'] = fmt_str
            ydl_opts['merge_output_format'] = 'mp4'
            
        elif dtype == 'audio':
            br = opts_data['bitrate']
            audio_fmt = opts_data.get('audio_fmt', 'mp3')
            
            ydl_opts['format'] = 'bestaudio/best'
            
            if audio_fmt == 'wav':
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }]
            else:  # mp3
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': br,
                }]
        
        # Handle playlists
        if is_playlist:
            ydl_opts['noplaylist'] = False
        else:
            ydl_opts['noplaylist'] = True
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        _open_folder(folder)
        
        # Return success message with track count if playlist
        if is_playlist:
            return {'success': True, 'message': 'Playlist downloaded successfully!'}
        else:
            return {'success': True, 'message': 'Downloaded successfully!'}
            
    except Exception as e: 
        return {'success': False, 'error': str(e)}
