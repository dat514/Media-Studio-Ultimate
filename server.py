import http.server
import socketserver
import urllib.parse
import os
import shutil
from ui import HTML_CONTENT

PORT = 8000

class FullPathHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/stream?path='):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if 'path' in params:
                fpath = params['path'][0]
                if os.path.exists(fpath):
                    self.send_response(200)
                    if fpath.lower().endswith('.mp4'): ctype = 'video/mp4'
                    elif fpath.lower().endswith('.png'): ctype = 'image/png'
                    elif fpath.lower().endswith('.jpg') or fpath.lower().endswith('.jpeg'): ctype = 'image/jpeg'
                    elif fpath.lower().endswith('.gif'): ctype = 'image/gif'
                    else: ctype = 'application/octet-stream'
                    
                    self.send_header('Content-type', ctype)
                    self.end_headers()
                    with open(fpath, 'rb') as f:
                        shutil.copyfileobj(f, self.wfile)
                    return
        
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_error(404)

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), FullPathHandler) as httpd:
        print(f"Serving at 127.0.0.1:{PORT}")
        httpd.serve_forever()
