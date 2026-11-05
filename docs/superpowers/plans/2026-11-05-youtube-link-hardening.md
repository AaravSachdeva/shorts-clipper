# YouTube Link Download Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Context

The user asked for "give it a YouTube link, it downloads the video and makes shorts". **This feature already exists:**

- `main.py` takes a positional `url` ("YouTube URL, file:// URL, or local file path").
- `--mode local`: `shorts_generator/local/downloader.py:download_youtube_local` downloads it with yt-dlp to `output/source_<id>.mp4`, writes `source_<id>.info.json` (used for the folder title), and reuses the cached file on reruns.
- `--mode api` (the default): `shorts_generator/downloader.py:download_youtube` sends the link to MuAPI `/youtube-download`.

Working command today:

```bash
.venv/bin/python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --mode local --profile cpu-slow
```

**Gap found (measured 2026-11-05, dry run with `yt-dlp --simulate` against a public video):** the installed yt-dlp 2026.10.26 enables only the `deno` JS runtime by default (`YoutubeDL.py:740`: `{'deno': {}}`). Deno is not installed. Node v24.21.0 is installed, but yt-dlp ignores it unless enabled, and the `yt_dlp_ejs` challenge-solver package is missing. Result: `WARNING: ... n challenge solving failed: Some formats may be missing`. Downloads work now but are degraded, and YouTube is phasing out extraction without a JS runtime. So the only work is to make link downloads reliable on this machine.

**Goal:** Make link-to-shorts reliable from the CLI (Task 1), and let the hosted review dashboard start it from a pasted YouTube link (Task 2).

**Architecture:** Task 1 adds one `js_runtimes` entry to the existing yt-dlp options and the solver package to the requirements. Task 2 adds a link form to `review.py` that starts `main.py` as one background subprocess at a time, and shows its status on the page.

**Tech Stack:** Python 3.11, yt-dlp 2026.10.26, Node 24, pytest.

**Spec:** this file's Context section (the user request + the measured gap).

## Global Constraints

- Keep `deno` enabled. Add `node` alongside it: `{"deno": {}, "node": {}}`.
- Never use yt-dlp `remote_components` (it downloads solver code from GitHub at run time). Install the solver as a pip package instead.
- Package name `yt-dlp-ejs` is **unverified** (from memory, observation 0002). Step 4 confirms it on PyPI before writing it into requirements.

## Review Focus

1. Dashboard link field gets a local path, `file://` or `--flag` (`download_youtube_local` accepts local files): it must be rejected with 400. Task 2 `test_youtube_url`.
2. Two links submitted back to back from the phone: only one pipeline runs, and the second gets 409. Task 2 `test_generate_starts_one_job`.
3. Pipeline fails (private video, no speech): the dashboard shows "Failed" and the escaped log tail, not a silent nothing. Task 2 `test_failed_job_shows_log_tail`.
4. Neither deno nor node on the PATH: yt-dlp still works with the degraded warning. Task 1 keeps every other option unchanged.
5. Dashboard restarted mid-run: the job status is forgotten, but the shorts still appear under Pending when done. This is accepted as a shortcut with a code comment.

---

### Task 1: Enable Node for yt-dlp and install the solver

**Files:**
- Modify: `shorts_generator/local/downloader.py` (the `ydl_opts` dict in `download_youtube_local`)
- Modify: `requirements-local.txt`
- Modify: `README.md` (the local-mode setup section)
- Test: `tests/test_downloader.py`

**Interfaces:**
- Consumes: `download_youtube_local(video_url: str, fmt: str = "720", out_dir: Optional[str] = None) -> str` (unchanged signature)
- Produces: nothing new

- [ ] **Step 1: Write the failing test** `test_download_enables_node_runtime(tmp_path, monkeypatch)` in `tests/test_downloader.py`. Monkeypatch `shorts_generator.local.downloader._import_ytdlp` to return a fake module whose `YoutubeDL(opts)` records `opts`, works as a context manager, and whose `extract_info` returns `{"id": "abc", "ext": "mp4"}` and `prepare_filename` returns a path it created under `tmp_path`. Call `download_youtube_local("https://www.youtube.com/watch?v=abc", out_dir=str(tmp_path))`. Assert:

```python
assert opts["js_runtimes"] == {"deno": {}, "node": {}}
assert opts["writeinfojson"] is True  # existing options kept
assert "remote_components" not in opts
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_downloader.py::test_download_enables_node_runtime -v`. Expected: FAIL with `KeyError: 'js_runtimes'`.

- [ ] **Step 3:** In `ydl_opts`, add `"js_runtimes": {"deno": {}, "node": {}},` with the one-line comment `# yt-dlp enables only deno by default; node is what most machines have`.

- [ ] **Step 4:** Run `.venv/bin/python -m pip index versions yt-dlp-ejs`. Expected: a version list. If it fails, use `yt-dlp[default]` instead (the extra that bundles the solver). Replace `yt-dlp>=2024.1.1` in `requirements-local.txt` with the confirmed form, keeping a version floor: `yt-dlp[default]>=2025.11.12` (or `yt-dlp>=…` plus `yt-dlp-ejs`). Then run `.venv/bin/python -m pip install -r requirements-local.txt`.

- [ ] **Step 5:** Run `.venv/bin/python -m pytest -q`. Expected: all tests pass.

- [ ] **Step 6: Real check.** Run:
  `.venv/bin/python -m yt_dlp --no-cache-dir --simulate --js-runtimes node --print "%(format_id)s %(height)s" "https://www.youtube.com/watch?v=jNQXAC9IVRw"`
  Expected: a format line and **no** `n challenge solving failed` warning. Then run the real pipeline once on a short public video: `.venv/bin/python main.py "<url>" --mode local --num-clips 1`. Expected: `output/source_<id>.mp4`, `output/source_<id>.info.json`, and one short in `output/<video title>/`.

- [ ] **Step 7:** In `README.md` local setup, add one line: local mode downloads YouTube links with yt-dlp and needs Node or Deno on the PATH, and show the `main.py "<youtube url>" --mode local` command.

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/local/downloader.py tests/test_downloader.py requirements-local.txt README.md
git commit -m "fix: let yt-dlp solve YouTube JS challenges with Node"
```

### Task 2: "Make shorts from a YouTube link" on the hosted review dashboard

User asked (2026-11-05): add the same link-to-shorts flow to the hosted UI. The hosted UI is `review.py`, the token-login mobile dashboard. It uses stdlib `http.server` and serves `LOCAL_OUTPUT_DIR`. A pipeline run takes from minutes to hours, so the dashboard starts `main.py` as a background subprocess and shows its status. It never runs the pipeline inside the request.

**Files:**
- Modify: `review.py`
- Modify: `README.md` (the review dashboard section)
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `shorts_generator.local.downloader._extract_youtube_video_id(source: str) -> Optional[str]`. Import it inside the function, the same way `main()` imports config lazily.
- Produces (all in `review.py`):
  - `youtube_url(text: str) -> Optional[str]`: returns `https://www.youtube.com/watch?v=<id>` only when the scheme is http/https, the id comes from `_extract_youtube_video_id`, and the id fully matches `[A-Za-z0-9_-]{11}`. Otherwise returns None. This is the trust boundary: it rejects local paths, `file://`, leading `-` and any extra query parameters.
  - `start_job(srv, url: str) -> bool`: returns False if `srv.job` is still running (`proc.poll() is None`). Otherwise it truncates and opens `srv.root / ".generate.log"`, runs `subprocess.Popen([*srv.cmd, "--", url], stdout=log, stderr=subprocess.STDOUT)` with no shell, inheriting the cwd, sets `srv.job = {"url", "proc", "started": time.time()}`, and returns True.
  - `make_server(root, token, host, port, pipeline_args=("--mode", "local"))` sets `srv.cmd = [sys.executable, str(Path(__file__).with_name("main.py")), *pipeline_args]` and `srv.job = None`.
  - `main()` gets a new flag `--pipeline-args` (default `"--mode local"`, split with `shlex.split`), so the operator can pass e.g. `"--mode local --profile cpu-slow"`.
  - New route `POST /generate`, form field `url`, cookie auth as for `/move`. It reuses the existing body-length check (`0 < length <= 4096`). It answers 400 `Not a YouTube video link` when `youtube_url` returns None, 409 `Already making shorts; wait for it to finish` when `start_job` returns False, and otherwise 303 to `/`.
  - `_page(groups, status, job=None)`: a form at the top of the page with `<input type="url" name="url" required placeholder="YouTube link">` and a "Make shorts" button posting to `/generate`, then one status line, with every value passed through `html.escape`:
    - running: `Making shorts from <url>, started HH:MM. Refresh to check.`
    - exit code 0: `Done: <url>. New shorts are under Pending.`
    - nonzero exit code: `Failed: <url>`, followed by the last 5 lines of `.generate.log` in a `<pre>`.

- [ ] **Step 1: Write the failing tests** in `tests/test_review.py`:

```python
@pytest.mark.parametrize("text,expected", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw&list=x", "https://www.youtube.com/watch?v=jNQXAC9IVRw"),
    ("https://youtu.be/jNQXAC9IVRw", "https://www.youtube.com/watch?v=jNQXAC9IVRw"),
    ("https://www.youtube.com/shorts/jNQXAC9IVRw", "https://www.youtube.com/watch?v=jNQXAC9IVRw"),
    ("/etc/passwd", None), ("file:///etc/passwd", None), ("--help", None),
    ("https://vimeo.com/123", None), ("https://youtu.be/short", None), ("", None),
])
def test_youtube_url(text, expected): ...

def test_generate_starts_one_job(root, server):
    # server.cmd = [sys.executable, "-c", "import sys,time;print(sys.argv[1:]);time.sleep(2)"]
    # POST /generate url=https://youtu.be/jNQXAC9IVRw -> 303
    # immediate second POST -> 409
    # GET / -> body contains b"Making shorts from" and the canonical url
    # after server.job["proc"].wait(): root/".generate.log" contains "'--', 'https://www.youtube.com/watch?v=jNQXAC9IVRw'"
    # GET / -> b"Done:"

def test_generate_rejects_bad_link_and_needs_cookie(server):
    # url=/etc/passwd -> 400 and server.job is None; no cookie -> 401

def test_failed_job_shows_log_tail(root, server):
    # server.cmd = [sys.executable, "-c", "print('<boom>');raise SystemExit(3)"]; POST, wait
    # GET / -> b"Failed:" and b"&lt;boom&gt;" in body, b"<boom>" not in body
```

  `list_groups` has to ignore `.generate.log`. It already skips plain files, and `test_list_groups_groups_by_video_folder` still passes.

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_review.py -v`. Expected: the new tests fail with `ImportError: cannot import name 'youtube_url'`.

- [ ] **Step 3:** Implement the Interfaces above in `review.py`. Put the form in the existing CSS style: the input goes full width, reusing the existing `form` and `button` rules, with `min-height:44px` on the input. Add the comment `# shortcut: job state lives in memory, a dashboard restart forgets it (the pipeline keeps running); persist it if that bites`.

- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q`. Expected: all tests pass.

- [ ] **Step 5: Real check.** Run `REVIEW_TOKEN=t .venv/bin/python review.py --port 8765 --pipeline-args "--mode local --num-clips 1"`. Open `http://localhost:8765/?token=t`, paste a short public YouTube link and submit. Expected: the status line reads "Making shorts…". After the run, "Done", and the new short shows under Pending. Check it at 375px width as well: no horizontal scroll.

- [ ] **Step 6:** In the `README.md` dashboard section, add 2 lines: paste a YouTube link to make shorts, one at a time, and use `--pipeline-args` to pass flags such as `--profile cpu-slow`.

- [ ] **Step 7: Commit**

```bash
git add review.py tests/test_review.py README.md
git commit -m "feat: make shorts from a YouTube link in the review dashboard"
```

## Not done

- `research/benchmark.py` creates its own `YoutubeDL`. It only reads channel listings, so it is left unchanged.
- Private or age-restricted videos (these need cookies) are not handled.
- API mode needs no change, because MuAPI does that download.
- At execution, copy this plan to `docs/superpowers/plans/2026-11-05-youtube-link-hardening.md`. Plan mode allowed only this file.
