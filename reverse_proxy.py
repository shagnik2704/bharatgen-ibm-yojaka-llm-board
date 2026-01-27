#!/usr/bin/env python3
"""
Simple reverse proxy to route different paths to different services.
Run this on port 9000, then expose it via ngrok.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import http.client
import sys

# Service mappings
SERVICES = {
    '/sanskrit': 'localhost:10008',
    '/spokentutorial': 'localhost:5000',
    '/llmboard': 'localhost:8002',
    '/kef': 'localhost:8003',
}

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy_request()
    
    def do_POST(self):
        self._proxy_request()
    
    def do_PUT(self):
        self._proxy_request()
    
    def do_DELETE(self):
        self._proxy_request()
    
    def do_PATCH(self):
        self._proxy_request()
    
    def _proxy_request(self):
        # Find matching service
        target = None
        path = self.path
        
        for prefix, service_addr in SERVICES.items():
            if path.startswith(prefix):
                target = service_addr
                # Remove the prefix from the path
                path = path[len(prefix):] if len(path) > len(prefix) else '/'
                break
        
        # Default to llmboard if no match
        if not target:
            target = 'localhost:8002'
        
        try:
            # Parse the target
            host, port = target.split(':')
            port = int(port)
            
            # Create connection to target service
            conn = http.client.HTTPConnection(host, port, timeout=30)
            
            # Read request body if present
            content_length = self.headers.get('Content-Length')
            body = None
            if content_length:
                body = self.rfile.read(int(content_length))
            
            # Forward headers (excluding hop-by-hop headers)
            headers = {}
            for key, value in self.headers.items():
                if key.lower() not in ['host', 'connection', 'content-length']:
                    headers[key] = value
            
            # Make request to target
            conn.request(self.command, path, body, headers)
            response = conn.getresponse()
            
            # Send response back to client
            self.send_response(response.status)
            
            # Forward response headers
            for header, value in response.getheaders():
                if header.lower() not in ['connection', 'transfer-encoding']:
                    self.send_header(header, value)
            self.end_headers()
            
            # Forward response body
            self.wfile.write(response.read())
            conn.close()
            
        except Exception as e:
            self.send_error(502, f"Proxy error: {str(e)}")
    
    def log_message(self, format, *args):
        # Custom logging
        print(f"[{self.address_string()}] {format % args}")

if __name__ == '__main__':
    port = 9000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    server = HTTPServer(('0.0.0.0', port), ProxyHandler)
    print(f"Reverse proxy running on port {port}")
    print(f"Routes:")
    for prefix, service in SERVICES.items():
        print(f"  {prefix} -> {service}")
    print(f"\nExpose this via ngrok: ngrok http {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
