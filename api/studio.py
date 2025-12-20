import os
import json
import subprocess
import re
from .base import _ff, _open_folder

def get_media_info(src):
    """
    Returns duration and generic type (video/audio).
    """
    try:
        cmd = [_ff(), '-i', src]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', creationflags=0x08000000 if os.name == 'nt' else 0)
        output = res.stderr
        
        duration = 0.0
        match_dur = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', output)
        if match_dur:
            h, m, s = match_dur.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            
        has_video = 'Video:' in output
        return {'success': True, 'duration': duration, 'type': 'video' if has_video else 'audio'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def studio_process(json_args):
    """
    Main processor for Studio.
    Args (JSON):
        src: str
        folder: str (output folder)
        format: str (target format extension, e.g. 'mp3', 'mp4')
        trim: {start: float, end: float}
        fade: {in: bool, out: bool, duration: float}
        audio: {volume: float, speed: float} (volume 0.0-2.0, speed 0.5-2.0)
    """
    try:
        args = json.loads(json_args)
        src = args['src']
        out_folder = args['folder']
        fmt = args.get('format', 'mp3').lower()
        
        # Output filename
        name = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(out_folder, f"{name}_studio.{fmt}")
        
        # 1. Base Command
        cmd = [_ff(), '-y']
        
        # 2. Input seeking (Faster trimming)
        # However, filtering requires re-encoding, so accurate seeking is needed.
        # We will use -ss before -i for speed, but since we are re-encoding, it acts as a seek.
        trim = args.get('trim')
        duration = 0
        if trim:
             start = float(trim.get('start', 0))
             end = float(trim.get('end', 0))
             duration = end - start
             if start > 0:
                 cmd.extend(['-ss', str(start)])
             if end > 0 and duration > 0:
                 cmd.extend(['-to', str(end)])
        
        cmd.extend(['-i', src])
        
        # 3. Filters
        # Speed changes duration, so Fade Out calculation is tricky if done in one pass.
        # Ideally: Trim -> Speed -> Volume -> Fade
        
        audio_opts = args.get('audio', {})
        speed = float(audio_opts.get('speed', 1.0))
        volume = float(audio_opts.get('volume', 1.0))
        
        fade = args.get('fade', {})
        fade_dur = float(fade.get('duration', 3.0))
        
        # Build Filter Complex
        filter_a = []
        filter_v = []
        
        # A. Speed
        if speed != 1.0:
            filter_a.append(f"atempo={speed}")
            filter_v.append(f"setpts={1/speed}*PTS")
            # If duration was set by trim, the output duration changes!
            if duration > 0: duration = duration / speed
            
        # B. Volume
        if volume != 1.0:
             filter_a.append(f"volume={volume}")
             
        # C. Fade
        # Fade In is easy (start at 0)
        if fade.get('in'):
            filter_a.append(f"afade=t=in:ss=0:d={fade_dur}")
            if args.get('type') == 'video':
                filter_v.append(f"fade=t=in:st=0:d={fade_dur}")
                
        # Fade Out is hard because we need strict duration.
        # Providing we calculated 'duration' correctly from trim and speed:
        if fade.get('out') and duration > 0:
            st = max(0, duration - fade_dur)
            filter_a.append(f"afade=t=out:st={st}:d={fade_dur}")
            if args.get('type') == 'video':
                filter_v.append(f"fade=t=out:st={st}:d={fade_dur}")

        # Assemble Filters
        if filter_a or filter_v:
             # If re-encoding is needed (it IS needed for effects)
             if filter_v and args.get('type') == 'video':
                 cmd.extend(['-vf', ",".join(filter_v)])
             if filter_a:
                 cmd.extend(['-af', ",".join(filter_a)])
        
        # 4. Encoders (Generic defaults)
        # If speed/volume/fade used, we must re-encode.
        if fmt in ['mp4', 'mkv', 'mov']:
            # Video Output
            pass # Default ffmpeg encoder usually fine
        elif fmt in ['mp3', 'wav', 'm4a', 'flac', 'm4r']:
            # Audio Output
            pass
            
        cmd.append(out_path)
        
        subprocess.run(cmd, check=True, creationflags=0x08000000 if os.name == 'nt' else 0)
        _open_folder(out_folder)
        return {'success': True}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
