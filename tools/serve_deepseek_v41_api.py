# SPDX-License-Identifier: Apache-2.0
"""Expose only authenticated OpenAI inference routes from a loopback model service."""
import argparse
import hmac
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROUTES = {"GET": {"/v1/models", "/health"},
          "POST": {"/v1/chat/completions", "/v1/completions"}}


def handler(upstream, key):
    target = urlsplit(upstream)
    if target.scheme != "http" or target.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("The public adapter requires a loopback HTTP model service")
    class InferenceHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def log_message(self, format, *args):
            pass
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()
        def do_OPTIONS(self):
            if urlsplit(self.path).path not in ROUTES["GET"] | ROUTES["POST"]:
                self.send_error(404)
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.end_headers()
        def do_GET(self):
            self.forward()
        def do_POST(self):
            self.forward()
        def forward(self):
            route = urlsplit(self.path).path
            if route not in ROUTES.get(self.command, set()):
                self.send_error(404)
                return
            supplied = self.headers.get("Authorization", "").encode()
            if route != "/health" and not hmac.compare_digest(supplied, ("Bearer " + key).encode()):
                self.send_error(401)
                return
            if self.headers.get("Transfer-Encoding"):
                self.send_error(400)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 <= length <= 64 * 2**20:
                self.send_error(413)
                return
            connection = http.client.HTTPConnection(target.hostname, target.port, timeout=1800)
            response_started = False
            try:
                body = self.rfile.read(length) if length else None
                connection.request(self.command, self.path, body=body, headers={
                    "Content-Type": self.headers.get("Content-Type", "application/json"),
                    "Authorization": "Bearer " + key, "Accept-Encoding": "identity"})
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                response_started = True
                while chunk := response.read1(65536):
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, http.client.HTTPException):
                if not response_started:
                    payload = json.dumps({"error": {"message": "Model service is starting or unavailable",
                                                    "type": "service_unavailable",
                                                    "code": "service_unavailable"}}).encode()
                    try:
                        self.send_response(503)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.send_header("Retry-After", "5")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.wfile.write(payload)
                        self.wfile.flush()
                    except OSError:
                        pass
                self.close_connection = True
            finally:
                connection.close()
    return InferenceHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:18552")
    parser.add_argument("--port", type=int, default=18553)
    parser.add_argument("--api-key-file", required=True, type=Path)
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip()
    if not key:
        raise ValueError("An API key is required")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.upstream, key))
    server.serve_forever()


if __name__ == "__main__":
    main()
