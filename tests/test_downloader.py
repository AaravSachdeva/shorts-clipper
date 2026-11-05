import json

from shorts_generator.local.downloader import group_dir, video_title


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


def test_group_dir_reuses_folder_for_same_source(tmp_path):
    src = _info(tmp_path, "My Talk")
    out = tmp_path / "out"
    first = group_dir(str(out), src)
    assert first == str(out / "My Talk")
    assert group_dir(str(out), src) == first


def test_group_dir_same_title_other_video_gets_suffix(tmp_path):
    out = tmp_path / "out"
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first = group_dir(str(out), _info(a, "My Talk"))
    second = group_dir(str(out), _info(b, "My Talk"))
    assert second == str(out / "My Talk (2)")
    assert first != second
    assert group_dir(str(out), str(b / "source_abc.mp4")) == second


def test_group_dir_never_adopts_unmarked_folder(tmp_path):
    out = tmp_path / "out"
    (out / "My Talk").mkdir(parents=True)  # made by hand or by an older run
    assert group_dir(str(out), _info(tmp_path, "My Talk")) == str(out / "My Talk (2)")
