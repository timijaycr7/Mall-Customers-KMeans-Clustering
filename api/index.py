from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {
            "project": "Mall Customers K-Means Clustering API",
            "endpoints": {
                "GET /api": "This help message",
                "GET /api/cluster": "Run clustering with auto-selected K (best silhouette score)",
                "GET /api/cluster?k=5": "Run clustering with a specific K (2-15)",
            },
            "source": "https://github.com/timijaycr7/Mall-Customers-KMeans-Clustering",
        }

        self.wfile.write(json.dumps(response, indent=2).encode())
