from shorts_generator.selection import gold as gd


def _words(tokens, t0=0.0, step=0.5):
    return [{"start": t0 + i * step, "end": t0 + i * step + 0.4, "word": w} for i, w in enumerate(tokens)]


SHORT = [f"w{i}" for i in range(30)]


def test_match_short_finds_span():
    filler = [f"x{i}" for i in range(200)]
    episode = _words(filler + SHORT + [f"y{i}" for i in range(200)])
    short = [t.upper() + "," for t in SHORT]
    short[12] = "changed"
    m = gd.match_short(episode, short)
    assert m is not None
    assert m["start"] == 100.0
    assert m["end"] == 100.0 + 29 * 0.5 + 0.4
    assert m["match_ratio"] >= 0.8


def test_match_short_rejects_unrelated():
    episode = _words([f"x{i}" for i in range(400)])
    assert gd.match_short(episode, SHORT) is None


def test_far_common_phrase_ignored():
    phrase = ["alpha", "beta", "gamma"]
    body = [f"b{i}" for i in range(27)]
    episode = _words(phrase + [f"x{i}" for i in range(1197)] + body)  # body starts at 600 s
    m = gd.match_short(episode, phrase + body)
    assert m["start"] == 600.0
    assert m["match_ratio"] == 0.9


def test_merge_keeps_hand_picks():
    old = [{"start": 1, "end": 20, "origin": "hand_pick", "rating": 5},
           {"start": 50, "end": 80, "origin": "official_short", "rating": 4}]
    new = [{"start": 90, "end": 120, "origin": "official_short", "rating": 4}]
    assert gd.merge_gold(old, new) == [old[0], new[0]]


def test_episode_gold_with_no_matches_is_empty_and_keeps_split(capsys):
    episode = _words([f"x{i}" for i in range(100)])
    shorts = [{"url": "u1", "views": 10, "tokens": SHORT}]
    out = gd.episode_gold("ep", "/v/ep.mp4", episode, shorts, old={"split": "test", "gold": []}, language="en")
    assert out == {"episode": "ep", "source": "/v/ep.mp4", "language": "en", "split": "test", "gold": []}
    assert "no official Shorts matched" in capsys.readouterr().out


def test_episode_gold_records_matches():
    episode = _words([f"x{i}" for i in range(200)] + SHORT)
    shorts = [{"url": "u1", "views": 1234, "tokens": SHORT}]
    out = gd.episode_gold("ep", "/v/ep.mp4", episode, shorts, old={}, language=None)
    [g] = out["gold"]
    assert g["origin"] == "official_short" and g["rating"] == 4 and g["views"] == 1234 and g["url"] == "u1"
    assert g["start"] == 100.0 and out["split"] in ("tune", "test")


class _Ydl:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.downloaded = []

    def extract_info(self, url, download=False):
        self.downloaded.append(download)
        return {"id": "abc", "ext": "m4a", "webpage_url": url, "title": "t", "view_count": 5}

    def prepare_filename(self, info):
        path = self.tmp_path / "abc.m4a"
        path.write_bytes(b"x")
        return str(path)


def test_short_entry_transcribes_the_audio(tmp_path, monkeypatch):
    from shorts_generator.local import transcriber
    seen = {}

    def fake_transcribe(path, language=None):
        seen["args"] = (path, language)
        return {"duration": 3.0, "segments": [{"start": 0, "end": 3, "text": "Hello world.", "words": [
            {"start": 0.0, "end": 0.5, "word": " Hello"}, {"start": 0.5, "end": 1.0, "word": " world."}]}]}

    monkeypatch.setattr(transcriber, "transcribe_local", fake_transcribe)
    ydl = _Ydl(tmp_path)
    entry = gd._short_entry(ydl, "u")
    assert entry == {"url": "u", "title": "t", "views": 5, "tokens": ["Hello", "world."]}
    assert ydl.downloaded == [True] and seen["args"] == (str(tmp_path / "abc.m4a"), "en")
