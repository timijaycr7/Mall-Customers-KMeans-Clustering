from http.server import BaseHTTPRequestHandler
import json
import os


def load_results():
    path = os.path.join(os.path.dirname(__file__), "..", "cluster_results.json")
    with open(path) as f:
        return json.load(f)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            results = load_results()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(results, indent=2).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
