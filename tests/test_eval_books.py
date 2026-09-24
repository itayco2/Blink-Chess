"""The pre-split book slices (plan section 1, "book slices"): dev.pgn and final.pgn, written once, hashed."""

import hashlib
import json

import pytest

from blink.eval import books

GAME = '[Event "?"]\r\n[Result "*"]\r\n\r\n1. {moves} *\r\n\r\n'
OPENINGS = ["e4 e5", "d4 d5", "c4 c5", "Nf3 Nf6", "g3 g6", "b3 b6", "e4 c5"]


def write_book(path, openings=OPENINGS):
    path.write_bytes("".join(GAME.format(moves=m) for m in openings).encode("utf-8"))


SMALL = {"dev": (1, 3), "final": (4, 7)}


def test_dev_and_final_book_slices_are_disjoint(tmp_path):
    source = tmp_path / "8moves_v3.pgn"
    write_book(source)
    manifest = books.write_slices(source, tmp_path / "out", slices=SMALL)
    dev = books.read_openings(tmp_path / "out" / "dev.pgn")
    final = books.read_openings(tmp_path / "out" / "final.pgn")
    assert [o.moves for o in dev] == [o.moves for o in books.read_openings(source, 1, 3)]
    assert [o.moves for o in final] == [o.moves for o in books.read_openings(source, 4, 4)]
    assert not {o.moves for o in dev} & {o.moves for o in final}
    assert manifest["dev"]["openings"] == 3 and manifest["final"]["openings"] == 4
    assert manifest["dev"]["first"] == 1 and manifest["final"]["first"] == 4


def test_the_slices_are_byte_exact_copies_of_the_source_games(tmp_path):
    source = tmp_path / "8moves_v3.pgn"
    write_book(source)
    books.write_slices(source, tmp_path / "out", slices=SMALL)
    joined = (tmp_path / "out" / "dev.pgn").read_bytes() + (tmp_path / "out" / "final.pgn").read_bytes()
    assert joined == source.read_bytes()


def test_the_manifest_records_each_files_sha256(tmp_path):
    source = tmp_path / "8moves_v3.pgn"
    write_book(source)
    books.write_slices(source, tmp_path / "out", slices=SMALL)
    manifest = json.loads((tmp_path / "out" / books.MANIFEST).read_text(encoding="utf-8"))
    for name in ("dev", "final"):
        data = (tmp_path / "out" / f"{name}.pgn").read_bytes()
        assert manifest[name]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_the_slices_are_written_once_and_a_changed_file_is_refused(tmp_path):
    source = tmp_path / "8moves_v3.pgn"
    write_book(source)
    first = books.write_slices(source, tmp_path / "out", slices=SMALL)
    assert books.write_slices(source, tmp_path / "out", slices=SMALL) == first
    (tmp_path / "out" / "dev.pgn").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        books.write_slices(source, tmp_path / "out", slices=SMALL)


def test_a_book_too_short_for_its_slices_is_refused(tmp_path):
    source = tmp_path / "8moves_v3.pgn"
    write_book(source, OPENINGS[:5])
    with pytest.raises(ValueError, match="7"):
        books.write_slices(source, tmp_path / "out", slices=SMALL)


def test_a_slice_name_resolves_to_its_own_file_once_it_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(books, "books_dir", lambda: tmp_path)
    assert books.resolve("final") == (tmp_path / books.BOOK_NAME, 10_001, 34_700)
    (tmp_path / "final.pgn").write_bytes(b"")
    assert books.resolve("final") == (tmp_path / "final.pgn", 1, 24_700)
    assert books.resolve("dev") == (tmp_path / books.BOOK_NAME, 1, 10_000)


REAL = books.books_dir() / books.MANIFEST


@pytest.mark.local
@pytest.mark.skipif(not REAL.is_file(), reason="the real slices are written by `blink eval books`")
def test_the_real_slices_hold_10000_and_24700_openings_with_their_recorded_hashes():
    manifest = json.loads(REAL.read_text(encoding="utf-8"))
    assert (manifest["dev"]["openings"], manifest["final"]["openings"]) == (10_000, 24_700)
    for name in ("dev", "final"):
        data = (books.books_dir() / f"{name}.pgn").read_bytes()
        assert hashlib.sha256(data).hexdigest() == manifest[name]["sha256"]
    assert manifest["overlap_move_sequences"] == 0
    around_the_cut = books.read_openings(books.books_dir() / books.BOOK_NAME, 10_000, 2)
    last_dev = books.read_openings(books.books_dir() / "dev.pgn", 10_000, 1)[0]
    first_final = books.read_openings(books.books_dir() / "final.pgn", 1, 1)[0]
    assert (last_dev.moves, first_final.moves) == (around_the_cut[0].moves, around_the_cut[1].moves)
