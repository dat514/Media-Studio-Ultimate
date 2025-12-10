import os
import subprocess
from .base import _ff

def convert(src, fmt, folder):
    try:
        name = os.path.splitext(os.path.basename(src))[0]
        outfile = os.path.join(folder, f"{name}.{fmt}")
        subprocess.run([_ff(), '-y', '-i', src, outfile], check=True, creationflags=0x08000000)
        return {'success': True}
    except Exception as e: return {'success': False, 'error': str(e)}
