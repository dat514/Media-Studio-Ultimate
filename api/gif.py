import os
import subprocess
import json
from .base import _ff, _open_folder

def make_gif(json_args):
    try:
        args = json.loads(json_args)
        src = args['src']
        out_path = os.path.join(args['folder'], os.path.splitext(os.path.basename(src))[0] + ".gif")
        
        cmd = [_ff(), '-y']
        if float(args['start']) > 0: cmd.extend(['-ss', args['start']])
        if args['end']: cmd.extend(['-to', args['end']])
        
        cmd.extend(['-i', src])
        
        flt = []
        if args.get('crop') and ':' in args['crop']:
            flt.append(f"crop={args['crop']}")
            
        flt.append(f"fps={args['fps']},scale={args['width']}:-1:flags=lanczos")
        flt.append("split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse")
        
        cmd.extend(['-vf', ",".join(flt)])
        cmd.append(out_path)
        
        subprocess.run(cmd, check=True, creationflags=0x08000000)
        _open_folder(out_path)
        return {'success': True, 'path': out_path}
    except Exception as e: return {'success': False, 'error': str(e)}
