"""Serve the evaluator UI (Laptop 2):  python ui/serve.py [port]   (default 5500)
Then open http://localhost:5500 and enter the API URL of Laptop 1 (http://<laptop1-ip>:8000).
"""
import functools
import http.server
import sys
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent))
print(f"PathSnap UI on http://0.0.0.0:{port}")
http.server.ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()
