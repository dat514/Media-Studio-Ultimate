import http.server
import socketserver
import urllib.parse
import os
import shutil

PORT = 8000

class FullPathHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve local files securely via specific endpoint
        if self.path.startswith('/stream?path='):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if 'path' in params:
                fpath = params['path'][0]
                if os.path.exists(fpath):
                    self.send_response(200)
                    # Simple MIME type detection
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
        
        # Serve the app HTML
        if self.path in ['/', '/index.html']:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            # We will inject the global HTML content here. 
            # Ideally, the server should import the HTML or read it.
            # But to keep it decoupled, we can let the main thread set the content 
            # or we can import ui here. 
            # For this refactor, we will import ui.
            from ui import HTML_CONTENT
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_error(404)

def run_server():
    # Allow reuse address to prevent "Address already in use" on restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), FullPathHandler) as httpd:
        print(f"Serving at 127.0.0.1:{PORT}")
        httpd.serve_forever()
