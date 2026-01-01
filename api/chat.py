from http.server import BaseHTTPRequestHandler
import json
import os
from urllib.parse import parse_qs

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/chat':
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            # Get request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                
                # Simple response for now
                response = {
                    "reply": f"Hi! I'm Johnny. You asked: '{message}'. I'm still setting up my full responses, but I'm excited to chat with you!",
                    "suggestions": [
                        "What's your favorite project? 🚀",
                        "Tell me about your skills",
                        "What makes you unique?",
                        "What are you learning?"
                    ]
                }
                
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                error_response = {
                    "reply": f"Sorry, I encountered an error: {str(e)}",
                    "suggestions": []
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()