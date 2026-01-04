import yt_dlp
import json
import os
from .base import choose_folder, _open_folder
from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error as ID3Error
from mutagen.mp4 import MP4, MP4Cover
from mutagen.wave import WAVE
import io

def analyze(url):
    with yt_dlp.YoutubeDL({'quiet':True}) as ydl:
        return json.dumps(ydl.extract_info(url, download=False), ensure_ascii=False)

def process_cover_art(img_path):
    """
    Open image, resize if too large, and convert to JPEG/PNG bytes.
    Returns: (mime_type, data_bytes)
    """
    try:
        with Image.open(img_path) as img:
            # Convert to RGB to ensure compatibility (e.g. if RGBA and saving as JPEG)
            if img.mode != 'RGB' and img.mode != 'RGBA':
                img = img.convert('RGB')
                
            # Resize if too big (limit to 1000x1000 to save space, user requested auto-resize)
            max_size = (1000, 1000)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save to bytes
            output = io.BytesIO()
            # Default to JPEG for compatibility and size, unless it was PNG and we want to keep it
            # But user said "png and jpeg". Let's prefer JPEG for MP3/MP4 covers usually
            # unless alpha channel is important? MP3/MP4 covers are usually JPEG.
            # Let's check original format.
            fmt = img.format if img.format in ['JPEG', 'PNG'] else 'JPEG'
            if fmt == 'JPEG':
                img = img.convert('RGB') # JPEG doesn't support alpha
                
            img.save(output, format=fmt, quality=90)
            data = output.getvalue()
            mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
            return mime, data
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def embed_cover(file_path, cover_data, mime_type):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mp3':
            try:
                audio = MP3(file_path, ID3=ID3)
            except Exception:
                audio = MP3(file_path)
                
            try:
                audio.add_tags()
            except ID3Error:
                pass
            
            audio.tags.add(
                APIC(
                    encoding=3, # 3 is for utf-8
                    mime=mime_type,
                    type=3, # 3 is for the cover image
                    desc=u'Cover',
                    data=cover_data
                )
            )
            # Force ID3v2.3 for Windows Explorer compatibility
            audio.save(v2_version=3)
            
        elif ext in ['.mp4', '.m4a']:
            video = MP4(file_path)
            # MP4Cover.FORMAT_JPEG or MP4Cover.FORMAT_PNG
            covr_fmt = MP4Cover.FORMAT_JPEG if mime_type == 'image/jpeg' else MP4Cover.FORMAT_PNG
            video['covr'] = [MP4Cover(cover_data, imageformat=covr_fmt)]
            video.save()
            
        elif ext == '.wav':
            # WAV embedding can be tricky, using ID3 chunk via mutagen
            try:
                audio = WAVE(file_path)
                try:
                    audio.add_tags()
                except:
                    pass
                
                # mutagen.wave uses ID3 tags if available
                # Logic: access .tags which is ID3
                audio.tags.add(
                    APIC(
                        encoding=3,
                        mime=mime_type,
                        type=3,
                        desc=u'Cover',
                        data=cover_data
                    )
                )
                audio.save()
            except Exception as w_err:
                print(f"WAV Embed Error: {w_err}")

    except Exception as e:
        print(f"Failed to embed cover: {e}")

def download(url, opts_json):
    try:
        folder = choose_folder()
        if not folder: return {'success': False}
        opts_data = json.loads(opts_json)
        dtype = opts_data['type']
        is_playlist = opts_data.get('is_playlist', False)
        cover_art_path = opts_data.get('cover_art', None)
        
        ydl_opts = {
            'outtmpl': os.path.join(folder, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['default']}}
        }
        
        # Capture downloaded filenames
        downloaded_files = []
        def pp_hook(d):
            if d['status'] == 'finished':
                downloaded_files.append(d['filename'])
        
        ydl_opts['progress_hooks'] = [pp_hook]
        
        if dtype == 'video':
            res = opts_data['res']
            fps = opts_data['fps']
            
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
            else:  
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': br,
                }]
        
        if is_playlist:
            ydl_opts['noplaylist'] = False
        else:
            ydl_opts['noplaylist'] = True
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Embed Cover Art if requested
        if cover_art_path and os.path.exists(cover_art_path):
             mime, data = process_cover_art(cover_art_path)
             if mime and data:
                 for fpath in downloaded_files:
                     
                     base, ext = os.path.splitext(fpath)
                     final_path = fpath
                     
                     if dtype == 'audio':
                         target_ext = '.' + opts_data.get('audio_fmt', 'mp3')
                         possible_path = base + target_ext
                         if os.path.exists(possible_path):
                             final_path = possible_path
                     elif dtype == 'video':
                         # merge_output_format is mp4
                         possible_path = base + '.mp4'
                         if os.path.exists(possible_path):
                             final_path = possible_path
                             
                     if os.path.exists(final_path):
                         embed_cover(final_path, data, mime)
        
        _open_folder(folder)
        
        if is_playlist:
            return {'success': True, 'message': 'Playlist downloaded successfully!'}
        else:
            return {'success': True, 'message': 'Downloaded successfully!'}
            
    except Exception as e: 
        return {'success': False, 'error': str(e)}
