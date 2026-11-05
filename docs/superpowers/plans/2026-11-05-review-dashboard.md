# Review Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> On approval, copy this file to `docs/superpowers/plans/2026-11-05-review-dashboard.md` and commit it with Task 1.

## Context

The owner wants to review rendered shorts from a phone and approve or reject each one. The shorts from one long video should appear as one group, named after that video.

Today, `--mode local` writes `output/short_01.mp4` (plus `.json` and `.txt` sidecars) into one flat folder (`shorts_generator/local/clipper.py:359`). A second long video overwrites the first one's shorts, and nothing records which video a short came from. So grouping must start in the pipeline. The dashboard only reads that layout.

**Owner decisions (2026-11-05):**
- Access is from a phone on home Wi-Fi. A shared token in `.env` protects the dashboard.
- Approve and Reject move a short's three files into `approved/` or `rejected/` inside the video's group folder.

**My calls (the owner can override them):**
- Stdlib `http.server` in one file, `review.py`. It adds no dependency and needs no JS or build step.
- The group name is the YouTube title for URLs and the file stem for local files.
- "Move back" returns a decided short to pending, so a mistaken tap on the phone can be undone.
- A Download link on each card, for uploading from the phone.

**Goal:** `python review.py` serves a mobile-friendly page on the LAN. The page lists shorts grouped by source video, plays them, and moves each one to approved or rejected.

**Architecture:** The pipeline writes each run to `output/<video name>/`. A short's status is the folder it is in: the group root means pending, `approved/` means approved, `rejected/` means rejected. There is no database and no state file. `review.py` holds three pure functions (`list_groups`, `move_short`, `parse_range`) and a `BaseHTTPRequestHandler` that renders HTML with f-strings and `html.escape`.

**Tech Stack:** Python 3.10+ stdlib (`http.server`, `html`, `hmac`, `shutil`, `pathlib`), pytest. yt-dlp is already a dependency.

**Spec:** this plan's Context section. There is no separate design doc.

## Global Constraints

- No new dependencies in any `requirements*.txt`.
- Never delete a short. Moving is the only file operation the dashboard does.
- `REVIEW_TOKEN` is required. `review.py` exits with a clear message if it is unset. Compare tokens with `hmac.compare_digest`. Never log the token.
- Any path built from request input must resolve inside `LOCAL_OUTPUT_DIR`, or the request gets 400 or 404.
- API mode is out of scope: its clips are hosted MuAPI URLs, not local files. The README says so.
- Work on branch `review-dashboard`. Commits end with the attribution lines from the session's system reminder.
- Tests: `python -m pytest -q`.

**Facts about external software, from memory and not yet verified (observation 0002):**
- (a) yt-dlp's `writeinfojson: True` writes `<outtmpl stem>.info.json`, which here is `source_<id>.info.json`. Task 1 Step 3 checks this.
- (b) iOS Safari plays `<video>` only when the server answers byte-range requests with 206. Task 3 adds Range support, and the phone check in Verification confirms it.

## Review Focus

1. **Re-running the same video.** Pending `short_XX` files in the group are overwritten. Approved and rejected ones in subfolders survive. Approving the new `short_01` while `approved/short_01.mp4` already exists must save it as `short_01-2.mp4`, never overwrite. Tested in Task 2.
2. **Hostile form or URL input.** Inputs like `group=../..`, `name=../x`, `to=deleted`, or `/media/../../etc/passwd` must get 400 or 404 and touch nothing. Tested in Tasks 2 and 3.
3. **A phone seeking in a video.** `Range: bytes=-500` and an open-ended `bytes=N-` must return a correct 206. An out-of-range request must return 416. Tested in Tasks 2 and 3.
4. **A YouTube title with `/`, `:` or `?`.** It must become a valid single folder name. An empty title becomes `untitled`. Tested in Task 1.
5. **A request with no cookie, including a media URL.** It must return 401 and never serve the video. Tested in Task 3.

---

### Task 1: Group pipeline output by source video

**Files:**
- Modify: `shorts_generator/local/downloader.py`. Add `"writeinfojson": True` to `ydl_opts`, and add `video_title()`.
- Modify: `shorts_generator/pipeline.py:59`. Pass `out_dir=os.path.join(LOCAL_OUTPUT_DIR, video_title(source_path))` to `crop_highlights_local`.
- Modify: `README.md`. Change the Local mode output path to `./output/<video name>/short_01.mp4`.
- Test: `tests/test_downloader.py` (new)

**Interfaces:**
- Produces: `video_title(source_path: str) -> str`. It reads `title` from `os.path.splitext(source_path)[0] + ".info.json"` when that file exists, and otherwise uses the file stem. It replaces `[\\/:*?"<>|\x00-\x1f]` with `_`, strips spaces and dots from both ends, cuts the result to 100 characters, and returns `"untitled"` if nothing is left.
- `harness/run.py` stays on the flat `output/` folder (default `out_dir`). It is a test harness, not the user path.

- [ ] **Step 1: Write failing tests** in `tests/test_downloader.py`:
  - `test_video_title_reads_info_json(tmp_path)`: write `source_abc.mp4` and `source_abc.info.json` with `{"title": "My Talk"}`. Assert `video_title(str(tmp_path/"source_abc.mp4")) == "My Talk"`.
  - `test_video_title_falls_back_to_stem`: assert `video_title("/x/episode 12.mp4") == "episode 12"`.
  - `test_video_title_sanitizes`: the info.json title is `'a/b: c?'`. Assert the result is `"a_b_ c_"`. A title of `"  ..  "` gives `"untitled"`.
- [ ] **Step 2:** Run `python -m pytest tests/test_downloader.py -q`. Expected: it fails on the `ImportError` for `video_title`.
- [ ] **Step 3:** Check fact (a). Run `grep -rn "infojson" $(python -c "import yt_dlp,os;print(os.path.dirname(yt_dlp.__file__))")/YoutubeDL.py | head`. Confirm that the info JSON name comes from the outtmpl with `info.json` as the extension. If it does not, take the title from `info["title"]` after `extract_info` and write `<stem>.info.json` yourself.
- [ ] **Step 4:** Implement `video_title`, add `writeinfojson`, and change the pipeline call.
- [ ] **Step 5:** Run `python -m pytest -q`. Expected: all tests pass.
- [ ] **Step 6:** Commit with the message `feat: write each video's shorts to output/<video name>/`.

### Task 2: Dashboard core (`review.py` pure functions)

**Files:**
- Create: `review.py` (repo root, beside `main.py`). This task adds only the functions, not the server.
- Test: `tests/test_review.py` (new)

**Interfaces:**
- `STATUSES = ("pending", "approved", "rejected")`. Pending is the group root. The other two are the subfolders of the same name.
- `list_groups(root: Path) -> list[dict]`. It returns `[{"name": str, "mtime": float, "shorts": [{"name": "short_01", "status": str, "rel": "approved/short_01.mp4", "meta": dict}]}]`. Groups are the direct child folders of `root` that hold `short_*.mp4` at their top level or in `approved/` or `rejected/`. Files in `root` itself (source downloads, transcript caches) are ignored. Groups are sorted newest `mtime` first. `meta` is the parsed `.json` sidecar, or `{}` when the sidecar is missing or invalid.
- `move_short(root: Path, group: str, name: str, to: str) -> Path`. `group` must be the name of an existing direct child folder: no `/`, not `.` or `..`. `name` must match `^short_\d+(-\d+)?$`. `to` must be in `STATUSES`. Anything else raises `ValueError`. The function finds the short in whichever status folder holds it, or raises `FileNotFoundError`. It moves the `.mp4`, `.json` and `.txt` (whichever exist) with `shutil.move`. If the target name is taken, it adds a `-2`, `-3` … suffix. It returns the new `.mp4` path.
- `parse_range(header: str, size: int) -> tuple[int, int] | None`. It returns an inclusive `(start, end)`, or `None` when the range is invalid or cannot be satisfied.

- [ ] **Step 1: Write failing tests:**
  - `test_list_groups_groups_by_video_folder`: create `Video A/short_01.mp4`, `Video A/approved/short_02.mp4`, `Video B/short_01.mp4` and `source_x.mp4` at the root. Assert the group names are `{"Video A", "Video B"}` and the statuses in Video A are `{"short_01": "pending", "short_02": "approved"}`.
  - `test_move_moves_sidecars`: move all three files to `rejected`. Assert they are present in `rejected/` and absent from the group root.
  - `test_move_back_to_pending`: move a short from `approved` to `pending`. Assert it is back at the group root.
  - `test_move_collision_gets_suffix`: the target `approved/short_01.mp4` exists. Assert the move returns `.../approved/short_01-2.mp4` and the old file is unchanged.
  - `test_move_rejects_bad_input`: each of `("../x", "short_01", "approved")`, `("Video A", "../short_01", "approved")` and `("Video A", "short_01", "deleted")` raises `ValueError`.
  - `test_parse_range`: `("bytes=0-99",1000)` gives `(0,99)`. `("bytes=900-",1000)` gives `(900,999)`. `("bytes=-100",1000)` gives `(900,999)`. `("bytes=0-5000",1000)` gives `(0,999)`. `("bytes=1000-",1000)` gives `None`. `("junk",1000)` gives `None`.
- [ ] **Step 2:** Run `python -m pytest tests/test_review.py -q`. Expected: it fails on the `ImportError`.
- [ ] **Step 3:** Implement the three functions.
- [ ] **Step 4:** Run `python -m pytest -q`. Expected: all tests pass.
- [ ] **Step 5:** Commit with the message `feat: review dashboard file layout, moves and range parsing`.

### Task 3: HTTP server, token, mobile page

**Files:**
- Modify: `review.py`. Add `Handler(BaseHTTPRequestHandler)`, `make_server(root: Path, token: str, host: str, port: int) -> ThreadingHTTPServer` and `main()`.
- Modify: `.env.example`. Add `REVIEW_TOKEN=change-me   # long random string; open http://<host>:8765/?token=<it> once on your phone`.
- Modify: `README.md`. Add a "Review dashboard" section: run command, phone link, folder layout, local mode only, plain HTTP so home Wi-Fi only.
- Test: `tests/test_review.py`

**Interfaces:**
- CLI: `python review.py [--host 0.0.0.0] [--port 8765]`. The root is `shorts_generator.config.LOCAL_OUTPUT_DIR` and the token is `os.environ["REVIEW_TOKEN"]`.
- Auth: `GET /?token=<t>` with a matching token sets the cookie `review=<t>; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000` and returns 303 to `/`. Every other request without a matching cookie gets 401 with one line of text explaining how to open the link. `SameSite=Strict` is the CSRF protection for POST.
- `GET /?status=pending|approved|rejected` (default `pending`) returns the page:
  - **Layout:** tabs with counts. One `<section>` per group that has shorts in this status, with an `<h2>` holding the video name and its count.
  - **Each card:** `<video controls playsinline preload="none" src="/media/<group>/<rel>">`, then `meta.post.title` (fall back to `meta.title`), score, caption and hashtags as selectable text, and a Download link (`<a download>`).
  - **Actions:** Approve and Reject forms (`POST /move`) on pending shorts, and a "Move back" form on decided ones.
- `GET /media/<group>/<rel>`: URL-decode the path, resolve it, and require that it is inside `root` and ends in `.mp4`. Otherwise return 404. Answer a `Range` header with 206 and `Content-Range`, an unsatisfiable range with 416, and no range with 200. Always send `Accept-Ranges: bytes` and `Content-Type: video/mp4`.
- `POST /move` takes form fields `group`, `name`, `to` and `status`, the tab to return to. It calls `move_short`. Success returns 303 to `/?status=<status>`, `ValueError` returns 400, and `FileNotFoundError` returns 404.
- Page rules:
  - `<meta name="viewport" content="width=device-width,initial-scale=1">`.
  - A single column, max width 480px, and 16px side padding.
  - Buttons at least 44px tall.
  - Video `width:100%; aspect-ratio:9/16; max-height:70vh`.
  - Light and dark colours through `prefers-color-scheme`.
  - Every user string goes through `html.escape`. No JS.

- [ ] **Step 1: Write failing tests.** Start `make_server(tmp_root, "t", "127.0.0.1", 0)` in a daemon thread and use `urllib.request` with a no-redirect opener:
  - `test_requires_cookie`: `GET /` gives 401. `GET /media/Video A/short_01.mp4` with no cookie gives 401.
  - `test_token_login_sets_cookie`: `GET /?token=t` gives 303 and a `Set-Cookie` containing `HttpOnly` and `SameSite=Strict`. A wrong token gives 401.
  - `test_page_lists_group`: with the cookie, `GET /` gives 200, and the body contains `Video A` and `short_01`.
  - `test_media_range`: with the cookie and `Range: bytes=0-9`, the response is 206, `Content-Range: bytes 0-9/<size>`, and a 10-byte body.
  - `test_media_traversal`: `GET /media/..%2F..%2Fetc/passwd` gives 404.
  - `test_post_move`: with the cookie, `POST /move` with `to=approved` gives 303, and the file is now in `approved/`. `to=deleted` gives 400.
- [ ] **Step 2:** Run `python -m pytest tests/test_review.py -q`. Expected: the new tests fail.
- [ ] **Step 3:** Implement the handler, `make_server` and `main`.
- [ ] **Step 4:** Run `python -m pytest -q`. Expected: all tests pass.
- [ ] **Step 5:** Update `.env.example` and `README.md`.
- [ ] **Step 6:** Commit with the message `feat: mobile review dashboard with token login`.

## Verification

1. `python -m pytest -q` passes.
2. Run `python main.py <short YouTube URL> --mode local --num-clips 2` and check that `output/<video title>/short_01.mp4` exists. To check the layout without an LLM key, copy any mp4 to `output/Test Video/short_01.mp4` instead.
3. Run `REVIEW_TOKEN=xyz python review.py`, then open `http://<server LAN IP>:8765/?token=xyz` on the phone. Confirm that:
   - the video plays and seeks (this checks fact (b));
   - Approve moves the short to the Approved tab, and its files are in `approved/`;
   - Move back returns it to Pending;
   - the page has no sideways scroll.
4. Run `ss -ltnp | grep 8765` and confirm the port listens. If the phone cannot connect, open the port to the LAN (for example `sudo ufw allow from 192.168.0.0/16 to any port 8765`) and ask the owner first, because that changes the firewall.
