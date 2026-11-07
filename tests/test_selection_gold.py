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
    def __init__(self, info, fail_times=0):
        self.info, self.fail_times, self.opened = info, fail_times, 0

    def extract_info(self, url, download=False):
        return self.info

    def urlopen(self, url):
        import io
        import json as _json
        self.opened += 1
        if self.opened <= self.fail_times:
            raise OSError("HTTP Error 429: Too Many Requests")
        return io.BytesIO(_json.dumps({"events": [{"tStartMs": 0, "segs": [{"utf8": "hello world"}]}]}).encode())


_INFO = {"webpage_url": "u", "title": "t", "view_count": 5, "automatic_captions": {"en": [{"ext": "json3", "url": "c"}]}}


def test_short_entry_reads_captions():
    assert gd._short_entry(_Ydl(_INFO), "u") == {"url": "u", "title": "t", "views": 5, "tokens": ["hello", "world"]}


def test_failed_caption_download_is_not_an_empty_entry(monkeypatch):
    monkeypatch.setattr(gd.time, "sleep", lambda s: None)
    assert gd._short_entry(_Ydl(_INFO, fail_times=1), "u")["tokens"] == ["hello", "world"]
    assert gd._short_entry(_Ydl(_INFO, fail_times=5), "u") is None


def test_short_without_english_track_is_cached_empty():
    assert gd._short_entry(_Ydl({**_INFO, "automatic_captions": {}}), "u")["tokens"] == []
