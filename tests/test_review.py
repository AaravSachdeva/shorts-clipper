import json

import pytest

from review import list_groups, move_short, parse_range


def _short(folder, name="short_01", meta=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.mp4").write_bytes(b"0123456789" * 10)
    (folder / f"{name}.json").write_text(json.dumps(meta or {"title": name}))
    (folder / f"{name}.txt").write_text("title\n")


@pytest.fixture
def root(tmp_path):
    _short(tmp_path / "Video A")
    _short(tmp_path / "Video A" / "approved", "short_02")
    _short(tmp_path / "Video B")
    (tmp_path / "source_x.mp4").write_bytes(b"")
    return tmp_path


def test_list_groups_groups_by_video_folder(root):
    groups = {g["name"]: g for g in list_groups(root)}
    assert set(groups) == {"Video A", "Video B"}
    assert {s["name"]: s["status"] for s in groups["Video A"]["shorts"]} == {
        "short_01": "pending",
        "short_02": "approved",
    }
    assert groups["Video A"]["shorts"][0]["meta"] == {"title": "short_01"}


def test_move_moves_sidecars(root):
    move_short(root, "Video A", "short_01", "rejected")
    for ext in ("mp4", "json", "txt"):
        assert (root / "Video A" / "rejected" / f"short_01.{ext}").exists()
        assert not (root / "Video A" / f"short_01.{ext}").exists()


def test_move_back_to_pending(root):
    new = move_short(root, "Video A", "short_02", "pending")
    assert new == root / "Video A" / "short_02.mp4"
    assert not (root / "Video A" / "approved" / "short_02.mp4").exists()


def test_move_collision_gets_suffix(root):
    _short(root / "Video A" / "approved", "short_01")
    old = (root / "Video A" / "approved" / "short_01.mp4").read_bytes()
    (root / "Video A" / "short_01.mp4").write_bytes(b"new")
    new = move_short(root, "Video A", "short_01", "approved")
    assert new == root / "Video A" / "approved" / "short_01-2.mp4"
    assert new.read_bytes() == b"new"
    assert (root / "Video A" / "approved" / "short_01.mp4").read_bytes() == old
    assert (root / "Video A" / "approved" / "short_01-2.json").exists()


@pytest.mark.parametrize("args", [
    ("../x", "short_01", "approved"),
    ("..", "short_01", "approved"),
    ("Video A", "../short_01", "approved"),
    ("Video A", "short_01", "deleted"),
    ("Nope", "short_01", "approved"),
])
def test_move_rejects_bad_input(root, args):
    with pytest.raises(ValueError):
        move_short(root, *args)
    assert (root / "Video A" / "short_01.mp4").exists()


def test_move_missing_short(root):
    with pytest.raises(FileNotFoundError):
        move_short(root, "Video A", "short_09", "approved")


@pytest.mark.parametrize("header,expected", [
    ("bytes=0-99", (0, 99)),
    ("bytes=900-", (900, 999)),
    ("bytes=-100", (900, 999)),
    ("bytes=0-5000", (0, 999)),
    ("bytes=1000-", None),
    ("bytes=5-2", None),
    ("bytes=-0", None),
    ("junk", None),
])
def test_parse_range(header, expected):
    assert parse_range(header, 1000) == expected


# --- HTTP server -------------------------------------------------------------

import http.client
import threading
from urllib.parse import quote, urlencode

from review import make_server

COOKIE = {"Cookie": "review=t"}


@pytest.fixture
def server(root):
    srv = make_server(root, "t", "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(server, method, path, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp, data


def test_requires_cookie(server):
    assert _req(server, "GET", "/")[0] == 401
    assert _req(server, "GET", "/media/" + quote("Video A/short_01.mp4"))[0] == 401
    assert _req(server, "GET", "/", {"Cookie": "review=wrong"})[0] == 401


def test_token_login_sets_cookie(server):
    status, resp, _ = _req(server, "GET", "/?token=t")
    assert status == 303
    cookie = resp.getheader("Set-Cookie")
    assert "review=t" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert _req(server, "GET", "/?token=nope")[0] == 401


def test_page_lists_group(server):
    status, _, body = _req(server, "GET", "/", COOKIE)
    assert status == 200
    assert b"Video A" in body and b"short_01" in body and b"Video B" in body
    assert b'name="viewport"' in body
    _, _, approved = _req(server, "GET", "/?status=approved", COOKIE)
    assert b"short_02" in approved and b"Video B" not in approved


def test_page_escapes_titles(root, server):
    (root / "Video B" / "short_01.json").write_text(json.dumps({"post": {"title": "<script>x</script>"}}))
    _, _, body = _req(server, "GET", "/", COOKIE)
    assert b"<script>x" not in body and b"&lt;script&gt;x" in body


def test_media_range(root, server):
    path = "/media/" + quote("Video A/short_01.mp4")
    status, resp, body = _req(server, "GET", path, {**COOKIE, "Range": "bytes=0-9"})
    assert status == 206
    assert resp.getheader("Content-Range") == "bytes 0-9/100"
    assert body == b"0123456789"
    status, resp, body = _req(server, "GET", path, COOKIE)
    assert status == 200 and len(body) == 100 and resp.getheader("Accept-Ranges") == "bytes"
    assert _req(server, "GET", path, {**COOKIE, "Range": "bytes=100-"})[0] == 416


def test_media_traversal(server):
    assert _req(server, "GET", "/media/..%2F..%2Fetc/passwd", COOKIE)[0] == 404
    assert _req(server, "GET", "/media/" + quote("Video A/short_01.json"), COOKIE)[0] == 404


def test_post_move(root, server):
    form = {"group": "Video A", "name": "short_01", "to": "approved", "status": "pending"}
    hdrs = {**COOKIE, "Content-Type": "application/x-www-form-urlencoded"}
    status, resp, _ = _req(server, "POST", "/move", hdrs, urlencode(form))
    assert status == 303 and resp.getheader("Location") == "/?status=pending"
    assert (root / "Video A" / "approved" / "short_01.mp4").exists()
    assert _req(server, "POST", "/move", hdrs, urlencode({**form, "to": "deleted"}))[0] == 400
    assert _req(server, "POST", "/move", hdrs, urlencode({**form, "name": "short_09"}))[0] == 404
    assert _req(server, "POST", "/move", {"Content-Type": hdrs["Content-Type"]}, urlencode(form))[0] == 401


def test_move_uses_source_status_when_name_exists_twice(root):
    _short(root / "Video A" / "approved", "short_01")  # re-run left a pending and an approved short_01
    new = move_short(root, "Video A", "short_01", "pending", frm="approved")
    assert new == root / "Video A" / "short_01-2.mp4"
    assert not (root / "Video A" / "approved" / "short_01.mp4").exists()
    assert (root / "Video A" / "short_01.mp4").exists()
    with pytest.raises(ValueError):
        move_short(root, "Video A", "short_01", "pending", frm="bogus")


def test_post_move_back_from_approved_tab(root, server):
    _short(root / "Video A" / "approved", "short_01")
    form = {"group": "Video A", "name": "short_01", "to": "pending", "status": "approved"}
    hdrs = {**COOKIE, "Content-Type": "application/x-www-form-urlencoded"}
    assert _req(server, "POST", "/move", hdrs, urlencode(form))[0] == 303
    assert not (root / "Video A" / "approved" / "short_01.mp4").exists()


# --- make shorts from a YouTube link ------------------------------------------

import sys

from review import youtube_url

ZOO = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
FORM = {**COOKIE, "Content-Type": "application/x-www-form-urlencoded"}


@pytest.mark.parametrize("text,expected", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw&list=x", ZOO),
    ("https://youtu.be/jNQXAC9IVRw", ZOO),
    ("https://www.youtube.com/shorts/jNQXAC9IVRw", ZOO),
    ("  https://m.youtube.com/watch?v=jNQXAC9IVRw  ", ZOO),
    ("/etc/passwd", None),
    ("file:///etc/passwd", None),
    ("--help", None),
    ("https://vimeo.com/123", None),
    ("https://youtu.be/short", None),
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw;rm", None),
    ("", None),
])
def test_youtube_url(text, expected):
    assert youtube_url(text) == expected


def _generate(server, url):
    return _req(server, "POST", "/generate", FORM, urlencode({"url": url}))


def test_generate_starts_one_job(root, server):
    server.cmd = [sys.executable, "-c", "import sys,time;print(sys.argv[1:]);time.sleep(2)"]
    status, resp, _ = _generate(server, "https://youtu.be/jNQXAC9IVRw")
    assert status == 303 and resp.getheader("Location") == "/"
    assert _generate(server, ZOO)[0] == 409
    _, _, body = _req(server, "GET", "/", COOKIE)
    assert b"Making shorts from" in body and ZOO.encode() in body
    server.job["proc"].wait()
    assert f"'--', '{ZOO}'" in (root / ".generate.log").read_text()
    _, _, body = _req(server, "GET", "/", COOKIE)
    assert b"Done:" in body
    assert b"Video A" in body  # the log file is not listed as a video


def test_generate_rejects_bad_link_and_needs_cookie(server):
    assert _generate(server, "/etc/passwd")[0] == 400
    assert server.job is None
    no_cookie = {"Content-Type": FORM["Content-Type"]}
    assert _req(server, "POST", "/generate", no_cookie, urlencode({"url": ZOO}))[0] == 401
    assert server.job is None


def test_failed_job_shows_log_tail(server):
    server.cmd = [sys.executable, "-c", "print('<boom>');raise SystemExit(3)"]
    assert _generate(server, ZOO)[0] == 303
    server.job["proc"].wait()
    _, _, body = _req(server, "GET", "/", COOKIE)
    assert b"Failed:" in body and b"&lt;boom&gt;" in body and b"<boom>" not in body


def test_page_has_link_form(server):
    _, _, body = _req(server, "GET", "/", COOKIE)
    assert b'action="/generate"' in body and b'name="url"' in body
