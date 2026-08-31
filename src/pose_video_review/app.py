"""Serve the local Pose Video Review interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .state import ViewerState


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def parse_byte_range(value: str | None, file_size: int) -> tuple[int, int] | None:
    """Parse one HTTP byte range, returning inclusive start and end offsets."""
    if not value:
        return None
    units, separator, spec = value.partition("=")
    if units.strip().casefold() != "bytes" or not separator or "," in spec:
        raise ValueError("Unsupported byte range")
    start_text, separator, end_text = spec.strip().partition("-")
    if not separator:
        raise ValueError("Invalid byte range")
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    elif end_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("Invalid byte range")
        start, end = max(0, file_size - length), file_size - 1
    else:
        raise ValueError("Invalid byte range")
    end = min(end, file_size - 1)
    if start < 0 or start > end or start >= file_size:
        raise ValueError("Byte range is outside the file")
    return start, end


class ViewerHandler(BaseHTTPRequestHandler):
    state: ViewerState
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self.serve_static("index.html")
            elif parsed.path.startswith("/static/"):
                self.serve_static(unquote(parsed.path.removeprefix("/static/")))
            elif parsed.path == "/api/trials":
                self.send_json({"trials": self.state.trials()})
            elif parsed.path == "/api/trial":
                trial = query.get("id", [""])[0]
                entries = self.state.trial_entries(trial)
                if entries:
                    self.send_json({"id": trial, "entries": entries})
                else:
                    self.send_json({"error": "Unknown trial"}, HTTPStatus.NOT_FOUND)
            elif parsed.path == "/api/poses":
                self.send_json(self.state.poses(query.get("id", [""])[0]))
            elif parsed.path == "/media":
                self.serve_media(query)
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        try:
            if urlparse(self.path).path != "/api/offsets":
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1024 * 1024:
                raise ValueError("Invalid request body size.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be an object.")
            self.send_json(self.state.save_offsets(payload.get("offsets")))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_HEAD(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/media":
                self.serve_media(parse_qs(parsed.query), head_only=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def serve_static(self, relative: str) -> None:
        normalized = posixpath.normpath(relative).lstrip("/")
        path = (STATIC_ROOT / normalized).resolve()
        if path != STATIC_ROOT and STATIC_ROOT not in path.parents:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_media(self, query: dict[str, list[str]], head_only: bool = False) -> None:
        path = Path(self.state.entry(query.get("id", [""])[0])["videoPath"])
        file_size = path.stat().st_size
        try:
            requested = parse_byte_range(self.headers.get("Range"), file_size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        start, end = requested or (0, file_size - 1)
        status = HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK
        self.send_response(status)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="OpenCap session or collection folder (default: current directory)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.folder}...", flush=True)
    ViewerHandler.state = ViewerState(args.folder)
    server = ViewerServer((args.host, args.port), ViewerHandler)
    print(f"Pose Video Review: http://{args.host}:{args.port}")
    print(f"Source: {ViewerHandler.state.source}")
    print(f"Videos: {len(ViewerHandler.state.entries)} in {len(ViewerHandler.state.trials())} trials")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Pose Video Review.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
