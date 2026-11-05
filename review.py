"""Review dashboard: approve or reject rendered shorts from a phone.

Layout under LOCAL_OUTPUT_DIR (written by the local pipeline):
    <video name>/short_01.mp4 (+ .json, .txt)   pending
    <video name>/approved/short_01.mp4           approved
    <video name>/rejected/short_01.mp4           rejected

A short's status is the folder it sits in, so there is no state file.
"""
import argparse
import hmac
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

STATUSES = ("pending", "approved", "rejected")
SIDECARS = (".mp4", ".json", ".txt")
_NAME = re.compile(r"short_\d+(-\d+)?")
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
LOG = ".generate.log"
_JOB_LOCK = threading.Lock()  # requests run on threads; two taps must not start two pipelines


def _folder(group_dir: Path, status: str) -> Path:
    return group_dir if status == "pending" else group_dir / status


def _meta(mp4: Path) -> Dict:
    try:
        return json.loads(mp4.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def list_groups(root: Path) -> List[Dict]:
    """One entry per source-video folder that holds shorts, newest first."""
    groups = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        shorts = [
            {"name": mp4.stem, "status": s, "rel": mp4.relative_to(d).as_posix(), "meta": _meta(mp4)}
            for s in STATUSES
            for mp4 in sorted(_folder(d, s).glob("short_*.mp4"))
        ]
        if shorts:
            mtime = max((d / s["rel"]).stat().st_mtime for s in shorts)
            groups.append({"name": d.name, "mtime": mtime, "shorts": shorts})
    return sorted(groups, key=lambda g: g["mtime"], reverse=True)


def move_short(root: Path, group: str, name: str, to: str, frm: Optional[str] = None) -> Path:
    """Move a short's mp4 and sidecars into the folder for status `to`; never overwrites.

    `frm` names the status it is moving from; a re-run video can leave the same
    name in two status folders, so the caller's view decides which one moves.
    """
    root = Path(root)
    if group in ("", ".", "..") or "/" in group or "\\" in group or not (root / group).is_dir():
        raise ValueError(f"bad group: {group!r}")
    if not _NAME.fullmatch(name):
        raise ValueError(f"bad short name: {name!r}")
    if to not in STATUSES or (frm is not None and frm not in STATUSES):
        raise ValueError(f"bad status: {to!r} / {frm!r}")
    group_dir = root / group
    candidates = (frm,) if frm else STATUSES
    src_dir = next((_folder(group_dir, s) for s in candidates if (_folder(group_dir, s) / f"{name}.mp4").is_file()), None)
    if src_dir is None:
        raise FileNotFoundError(f"{group}/{name}")
    dest_dir = _folder(group_dir, to)
    if src_dir == dest_dir:
        return src_dir / f"{name}.mp4"
    dest_dir.mkdir(exist_ok=True)
    base, stem, n = re.sub(r"-\d+$", "", name), name, 2
    while any((dest_dir / f"{stem}{ext}").exists() for ext in SIDECARS):
        stem, n = f"{base}-{n}", n + 1
    for ext in SIDECARS:
        if (src_dir / f"{name}{ext}").exists():
            shutil.move(str(src_dir / f"{name}{ext}"), str(dest_dir / f"{stem}{ext}"))
    return dest_dir / f"{stem}.mp4"


def parse_range(header: str, size: int) -> Optional[Tuple[int, int]]:
    """Inclusive (start, end) for a single `bytes=` range, or None if invalid/unsatisfiable."""
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", (header or "").strip())
    if not m or not (m.group(1) or m.group(2)):
        return None
    if not m.group(1):  # suffix range: last N bytes
        n = int(m.group(2))
        return (max(size - n, 0), size - 1) if n and size else None
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    return (start, end) if start <= end and start < size else None


def youtube_url(text: str) -> Optional[str]:
    """Canonical watch URL for a YouTube video link, or None for anything else.

    The dashboard passes this to the pipeline, which also accepts local paths,
    so only a rebuilt https://www.youtube.com/watch?v=<id> ever gets through.
    """
    from shorts_generator.local.downloader import _extract_youtube_video_id

    text = (text or "").strip()
    if urlsplit(text).scheme not in ("http", "https"):
        return None
    video_id = _extract_youtube_video_id(text) or ""
    return f"https://www.youtube.com/watch?v={video_id}" if _VIDEO_ID.fullmatch(video_id) else None


def start_job(srv, url: str) -> bool:
    """Start the pipeline on `url` in the background; False if one is already running."""
    # shortcut: job state lives in memory, a dashboard restart forgets it (the pipeline keeps running); persist it if that bites
    with _JOB_LOCK:
        if srv.job and srv.job["proc"].poll() is None:
            return False
        with open(srv.root / LOG, "w", encoding="utf-8") as log:
            proc = subprocess.Popen([*srv.cmd, "--", url], stdout=log, stderr=subprocess.STDOUT)
        srv.job = {"url": url, "proc": proc, "started": time.time()}
        return True


def _job_html(job: Optional[Dict], root: Path) -> str:
    if not job:
        return ""
    e, url, rc = html.escape, job["url"], job["proc"].poll()
    if rc is None:
        started = time.strftime("%H:%M", time.localtime(job["started"]))
        return f'<p class="job">Making shorts from {e(url)}, started {started}. Refresh to check.</p>'
    if rc == 0:
        return f'<p class="job">Done: {e(url)}. New shorts are under Pending.</p>'
    try:
        tail = "\n".join((Path(root) / LOG).read_text(encoding="utf-8", errors="replace").splitlines()[-5:])
    except OSError:
        tail = ""
    return f'<p class="job fail">Failed: {e(url)}</p><pre>{e(tail)}</pre>'


# --- HTTP server -------------------------------------------------------------

CSS = """
:root{--bg:#f6f6f4;--card:#fff;--fg:#1b1b1b;--muted:#5f5f5f;--line:#dcdcd8;--ok:#1f7a3a;--no:#b3261e;--accent:#1f4fb3}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--card:#1e1e1e;--fg:#ececec;--muted:#a8a8a8;--line:#333;--ok:#4caf6a;--no:#ef6b62;--accent:#7aa2ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:480px;margin:0 auto;padding:12px 16px 48px}
h1{font-size:1.25rem;margin:8px 0 12px}
nav{display:flex;gap:8px;margin-bottom:16px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:1}
nav a{flex:1;text-align:center;padding:10px 4px;min-height:44px;border:1px solid var(--line);border-radius:10px;color:var(--fg);text-decoration:none}
nav a[aria-current=page]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
h2{font-size:1.05rem;margin:24px 0 8px;overflow-wrap:anywhere}
h2 small{color:var(--muted);font-weight:normal}
article{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:16px}
video{display:block;width:100%;aspect-ratio:9/16;max-height:70vh;background:#000;border-radius:10px}
h3{font-size:1rem;margin:10px 0 4px;overflow-wrap:anywhere}
p{margin:4px 0;overflow-wrap:anywhere}
.meta{color:var(--muted);font-size:.9rem}
.tags{color:var(--accent);font-size:.9rem}
form{display:flex;gap:8px;margin-top:10px}
button,.dl{flex:1;min-height:44px;font:inherit;font-weight:600;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);display:flex;align-items:center;justify-content:center;text-decoration:none}
.approve{background:var(--ok);border-color:var(--ok);color:#fff}
.reject{color:var(--no);border-color:var(--no)}
.empty{color:var(--muted)}
input{flex:2;min-width:0;min-height:44px;font:inherit;padding:0 10px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg)}
.job{color:var(--muted)}
.fail{color:var(--no)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;color:var(--muted)}
"""


def _page(groups: List[Dict], status: str, job_html: str = "") -> str:
    e = html.escape
    counts = {s: sum(sh["status"] == s for g in groups for sh in g["shorts"]) for s in STATUSES}
    tabs = "".join(
        f'<a href="/?status={s}"{" aria-current=page" if s == status else ""}>{s.title()} ({counts[s]})</a>'
        for s in STATUSES
    )
    body = []
    for g in groups:
        shorts = [sh for sh in g["shorts"] if sh["status"] == status]
        if not shorts:
            continue
        body.append(f'<section><h2>{e(g["name"])} <small>· {len(shorts)}</small></h2>')
        for sh in shorts:
            meta, post = sh["meta"], sh["meta"].get("post") or {}
            src = "/media/" + quote(f'{g["name"]}/{sh["rel"]}')
            title = post.get("title") or meta.get("title") or sh["name"]
            score = f'score {meta["score"]} · ' if meta.get("score") is not None else ""
            tags = " ".join(post.get("hashtags") or [])
            hidden = (f'<input type="hidden" name="group" value="{e(g["name"])}">'
                      f'<input type="hidden" name="name" value="{e(sh["name"])}">'
                      f'<input type="hidden" name="status" value="{status}">')
            if status == "pending":
                actions = ('<button class="approve" name="to" value="approved">Approve</button>'
                           '<button class="reject" name="to" value="rejected">Reject</button>')
            else:
                actions = '<button name="to" value="pending">Move back</button>'
            body.append(
                f'<article><video controls playsinline preload="none" src="{e(src)}"></video>'
                f'<h3>{e(str(title))}</h3><p class="meta">{e(score)}{e(sh["name"])}</p>'
                + (f'<p>{e(str(post["caption"]))}</p>' if post.get("caption") else "")
                + (f'<p class="tags">{e(tags)}</p>' if tags else "")
                + f'<form method="post" action="/move">{hidden}{actions}'
                f'<a class="dl" href="{e(src)}" download="{e(g["name"] + " - " + sh["name"])}.mp4">Download</a></form>'
                "</article>"
            )
        body.append("</section>")
    if not body:
        body.append(f'<p class="empty">Nothing {status}.</p>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Shorts Review</title><style>{CSS}</style></head>"
        f'<body><main><h1>Shorts Review</h1>'
        '<form method="post" action="/generate"><input type="url" name="url" required placeholder="YouTube link" aria-label="YouTube link">'
        '<button>Make shorts</button></form>'
        f'{job_html}<nav aria-label="Status">{tabs}</nav>{"".join(body)}</main></body></html>'
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "ShortsReview"

    def log_message(self, fmt, *args):  # never write the login token to the log
        sys.stderr.write(re.sub(r"token=[^&\s]*", "token=***", fmt % args) + "\n")

    def _send(self, code: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8", headers=None):
        self.send_response(code)
        for k, v in {"Content-Type": ctype, "Content-Length": str(len(body)), **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authed(self) -> bool:
        try:
            morsel = SimpleCookie(self.headers.get("Cookie", "")).get("review")
        except CookieError:
            return False
        return bool(morsel) and hmac.compare_digest(morsel.value.encode(), self.server.token.encode())

    def _deny(self):
        self._send(401, b"Not signed in. Open the dashboard link with ?token=<REVIEW_TOKEN> once on this device.\n")

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/" and "token" in query:
            if not hmac.compare_digest(query["token"][0].encode(), self.server.token.encode()):
                return self._deny()
            cookie = f"review={self.server.token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000"
            return self._send(303, headers={"Location": "/", "Set-Cookie": cookie})
        if not self._authed():
            return self._deny()
        if url.path == "/":
            status = query.get("status", ["pending"])[0]
            status = status if status in STATUSES else "pending"
            page = _page(list_groups(self.server.root), status, _job_html(self.server.job, self.server.root)).encode()
            return self._send(200, page, "text/html; charset=utf-8", {"Cache-Control": "no-store"})
        if url.path.startswith("/media/"):
            return self._media(unquote(url.path[len("/media/"):]))
        self._send(404, b"Not found\n")

    do_HEAD = do_GET

    def _media(self, rel: str):
        root = self.server.root.resolve()
        path = (root / rel).resolve()
        if not (path.is_relative_to(root) and path.suffix == ".mp4" and path.is_file()):
            return self._send(404, b"Not found\n")
        size = path.stat().st_size
        start, end, code = 0, size - 1, 200
        headers = {"Accept-Ranges": "bytes"}
        if self.headers.get("Range"):
            rng = parse_range(self.headers["Range"], size)
            if rng is None:
                return self._send(416, headers={**headers, "Content-Range": f"bytes */{size}"})
            (start, end), code = rng, 206
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(code)
        for k, v in {"Content-Type": "video/mp4", "Content-Length": str(end - start + 1), **headers}.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(left, 1 << 16))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # phones drop range requests mid-stream all the time

    def do_POST(self):
        if not self._authed():
            return self._deny()
        path = urlsplit(self.path).path
        if path not in ("/move", "/generate"):
            return self._send(404, b"Not found\n")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= 4096:
            return self._send(400, b"Bad form\n")
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8", "replace")).items()}
        if path == "/generate":
            url = youtube_url(form.get("url", ""))
            if not url:
                return self._send(400, b"Not a YouTube video link\n")
            if not start_job(self.server, url):
                return self._send(409, b"Already making shorts; wait for it to finish\n")
            return self._send(303, headers={"Location": "/"})
        back = form.get("status") if form.get("status") in STATUSES else "pending"
        try:
            move_short(self.server.root, form.get("group", ""), form.get("name", ""), form.get("to", ""), frm=back)
        except ValueError as err:
            return self._send(400, f"{err}\n".encode())
        except FileNotFoundError:
            return self._send(404, b"Short not found (already moved?)\n")
        self._send(303, headers={"Location": f"/?status={back}"})


def make_server(root: Path, token: str, host: str, port: int, pipeline_args=("--mode", "local")) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.root, srv.token = Path(root), token
    srv.cmd = [sys.executable, str(Path(__file__).with_name("main.py")), *pipeline_args]
    srv.job = None
    return srv


def main() -> int:
    parser = argparse.ArgumentParser(description="Review dashboard for rendered shorts")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0, the whole LAN)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pipeline-args", default="--mode local",
                        help='main.py flags for links pasted on the page (default: "--mode local")')
    args = parser.parse_args()

    from shorts_generator.config import LOCAL_OUTPUT_DIR  # also loads .env

    token = os.environ.get("REVIEW_TOKEN", "").strip()
    if not token:
        print("REVIEW_TOKEN is not set. Put a long random string in .env, e.g.\n"
              "  REVIEW_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')", file=sys.stderr)
        return 1
    root = Path(LOCAL_OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    srv = make_server(root, token, args.host, args.port, shlex.split(args.pipeline_args))
    print(f"Reviewing {root.resolve()}\nOn your phone open: http://<this machine's LAN IP>:{args.port}/?token=<REVIEW_TOKEN>", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
