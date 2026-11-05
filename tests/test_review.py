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
