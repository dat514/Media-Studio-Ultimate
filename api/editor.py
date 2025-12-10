import os
import subprocess
import json
from .base import _ff, _open_folder

def edit_media(json_args):
    try:
        args = json.loads(json_args)
        src = args['src']
        out_folder = args['folder']
        name = os.path.splitext(os.path.basename(src))[0]
        ext = os.path.splitext(src)[1] if args.get('crop') and args['crop']['shape']=='rect' else '.png' 
        if ext=='.mp4' or ext=='.avi': ext = '.mp4' 
        
        out = os.path.join(out_folder, f"{name}_edited{ext}")
        
        cmd = [_ff(), '-y', '-i', src]
        vf = []
        
        c = args.get('crop')
        if c:
            w,h,x,y = c['w'], c['h'], c['x'], c['y']
            if c['shape'] == 'rect':
                vf.append(f"crop={w}:{h}:{x}:{y}")
            elif c['shape'] == 'circle':
                vf.append(f"crop={w}:{h}:{x}:{y}")
                vf.append(f"format=yuva420p,geq=lum='p(X,Y)':a='if(lte((X-W/2)^2+(Y-H/2)^2, (min(W,H)/2)^2), 255, 0)'")
            elif c['shape'] == 'triangle':
                 vf.append(f"crop={w}:{h}:{x}:{y}")
                 vf.append(f"format=yuva420p,geq=lum='p(X,Y)':a='if(gt(Y, -2*H/W * abs(X-W/2) + H), 255, 0)'")

        rw, rh = args.get('w'), args.get('h')
        if rw and rh and int(rw)>0:
            vf.append(f"scale={rw}:{rh}")
            
        if vf: cmd.extend(['-vf', ",".join(vf)])
        
        cmd.append(out)
        subprocess.run(cmd, check=True, creationflags=0x08000000)
        _open_folder(out_folder)
        return {'success': True}
    except Exception as e: return {'success': False, 'error': str(e)}
