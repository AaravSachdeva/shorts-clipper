import json

from shorts_generator.local.downloader import video_title


def _info(tmp_path, title):
    (tmp_path / "source_abc.mp4").write_bytes(b"")
    (tmp_path / "source_abc.info.json").write_text(json.dumps({"title": title}))
    return str(tmp_path / "source_abc.mp4")


def test_video_title_reads_info_json(tmp_path):
    assert video_title(_info(tmp_path, "My Talk")) == "My Talk"


def test_video_title_falls_back_to_stem():
    assert video_title("/x/episode 12.mp4") == "episode 12"


def test_video_title_sanitizes(tmp_path):
    assert video_title(_info(tmp_path, "a/b: c?")) == "a_b_ c_"
    assert video_title(_info(tmp_path, "  ..  ")) == "untitled"
