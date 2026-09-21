#!/usr/bin/env python3
"""Local review server for benchmark compositions and QA items."""
from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
STATIC = ROOT / "static"
DATA = REPOSITORY / "benchmark" / "data"
AUDIO = REPOSITORY / "benchmark" / "audio"
ANNOTATIONS = ROOT / "data"

JSON_ROUTES = {
    "/api/compositions": DATA / "composition_manifest.private.json",
    "/api/qa": DATA / "qa.private.json",
    "/api/annotations/composition": ANNOTATIONS / "composition_annotations.json",
    "/api/annotations/qa": ANNOTATIONS / "qa_annotations.json",
}
STATIC_ROUTES = {
    "/": STATIC / "index.html",
    "/index.html": STATIC / "index.html",
    "/app.js": STATIC / "app.js",
    "/style.css": STATIC / "style.css",
}
WRITABLE_ROUTES = {
    "/api/annotations/composition": ANNOTATIONS / "composition_annotations.json",
    "/api/annotations/qa": ANNOTATIONS / "qa_annotations.json",
}


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "TemporalReview/1.0"

    def send_bytes(self, value: bytes, content_type: str, *, no_store: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(value)

    def send_audio(self, target: Path) -> None:
        size = target.stat().st_size
        start, end = 0, size - 1
        partial = False
        requested = self.headers.get("Range", "")
        if requested.startswith("bytes="):
            try:
                left, right = requested.removeprefix("bytes=").split("-", 1)
                start = int(left or 0)
                end = min(int(right) if right else end, end)
                if start < 0 or start > end:
                    raise ValueError
                partial = True
            except ValueError:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return

        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        try:
            with target.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in JSON_ROUTES:
            return self.send_bytes(
                JSON_ROUTES[path].read_bytes(),
                "application/json; charset=utf-8",
                no_store=True,
            )
        if path in STATIC_ROUTES:
            kind = {
                "/app.js": "text/javascript; charset=utf-8",
                "/style.css": "text/css; charset=utf-8",
            }.get(path, "text/html; charset=utf-8")
            return self.send_bytes(STATIC_ROUTES[path].read_bytes(), kind)
        if path.startswith("/audio/"):
            target = AUDIO / Path(unquote(path.removeprefix("/audio/"))).name
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            return self.send_audio(target)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        target = WRITABLE_ROUTES.get(path)
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 5 * 1024 * 1024:
                raise ValueError
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST)
            return

        encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(target)
        self.send_bytes(b'{"ok":true}\n', "application/json")

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"{self.address_string()} - {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Temporal benchmark review: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
