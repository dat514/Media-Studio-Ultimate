import urllib.request
import urllib.parse
import urllib.error
import json
import os
import time
from .base import _open_folder

def shorten_url(url, alias=None):
    try:
        api_url = f"https://is.gd/create.php?format=simple&url={urllib.parse.quote(url)}"
        if alias:
            api_url += f"&shorturl={urllib.parse.quote(alias)}"
        
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                short_url = response.read().decode('utf-8')
                return {'success': True, 'url': short_url}
            else:
                return {'success': False, 'error': f"HTTP {response.status}"}
    except urllib.error.HTTPError as e:
        # is.gd returns 400 etc for bad aliases
        if e.code == 400 or e.code == 500: # Try to read the error message
             try:
                 err_msg = e.read().decode('utf-8')
                 # is.gd simple format usually returns "Error: ..."
                 return {'success': False, 'error': err_msg}
             except:    
                 return {'success': False, 'error': f"HTTP Error {e.code}"}
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_qr_cleanup(src, folder):
    try:
        if not os.path.exists(src): return {'success': False, 'error': 'Source file not found'}
        name = os.path.basename(src)
        dst = os.path.join(folder, name)
        import shutil
        shutil.copy2(src, dst)
        os.remove(src)
        return {'success': True}
    except Exception as e: return {'success': False, 'error': str(e)}

def generate_advanced_qr(json_args):
    try:
        import qrcode
        from qrcode.image.styledpil import StyledPilImage
        from qrcode.image.styles.moduledrawers import SquareModuleDrawer, GappedSquareModuleDrawer, CircleModuleDrawer, RoundedModuleDrawer, VerticalBarsDrawer, HorizontalBarsDrawer
        from qrcode.image.styles.colormasks import SolidFillColorMask, RadialGradiantColorMask, SquareGradiantColorMask, HorizontalGradiantColorMask, VerticalGradiantColorMask
        from PIL import Image

        args = json.loads(json_args)
        data = args['data']
        if not data: return {'success': False, 'error': 'No data'}
        
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(data)
        qr.make(fit=True)
        
        # Drawer Map
        drawers = {
            'square': SquareModuleDrawer(), 'circle': CircleModuleDrawer(), 'rounded': RoundedModuleDrawer(),
            'gapped': GappedSquareModuleDrawer(), 'vertical': VerticalBarsDrawer(), 'horizontal': HorizontalBarsDrawer()
        }
        drawer = drawers.get(args.get('drawer'), SquareModuleDrawer())
        
        # Colors
        def hex2rgb(h): return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        back_c = hex2rgb(args.get('back_color', '#ffffff'))
        mask_type = args.get('mask', 'solid')
        
        if mask_type == 'solid':
            front_c = hex2rgb(args.get('fill_color', '#000000'))
            mask = SolidFillColorMask(front_color=front_c, back_color=back_c)
        else:
            c_start = hex2rgb(args.get('gradient_start', '#000000'))
            c_end = hex2rgb(args.get('gradient_end', '#000000'))
            
            if mask_type == 'radial':
                mask = RadialGradiantColorMask(back_color=back_c, center_color=c_start, edge_color=c_end)
            elif mask_type == 'square':
                mask = SquareGradiantColorMask(back_color=back_c, center_color=c_start, edge_color=c_end)
            elif mask_type == 'horizontal':
                mask = HorizontalGradiantColorMask(back_color=back_c, left_color=c_start, right_color=c_end)
            elif mask_type == 'vertical':
                mask = VerticalGradiantColorMask(back_color=back_c, top_color=c_start, bottom_color=c_end)
            else:
                mask = SolidFillColorMask(front_color=(0,0,0), back_color=back_c)

        img = qr.make_image(image_factory=StyledPilImage, module_drawer=drawer, color_mask=mask)
        
        # Logo Embed
        logo_path = args.get('logo_path')
        if logo_path and os.path.exists(logo_path):
            img_w, img_h = img.size
            logo = Image.open(logo_path)
            
            # Resize to 25% of QR width
            logo_max = int(img_w * 0.25)
            logo.thumbnail((logo_max, logo_max), Image.Resampling.LANCZOS)
            
            pos = ((img_w - logo.size[0]) // 2, (img_h - logo.size[1]) // 2)
            
            # If image is not RGBA, convert it
            img = img.convert("RGBA")
            if logo.mode != 'RGBA': logo = logo.convert("RGBA")
            
            img.paste(logo, pos, mask=logo)
        
        # Save
        temp_dir = os.path.join(os.getcwd(), 'screenshots')
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        
        fname = f"qr_{int(time.time()*1000)}.png"
        out_path = os.path.join(temp_dir, fname)
        img.save(out_path)
        
        return {'success': True, 'path': out_path}
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
