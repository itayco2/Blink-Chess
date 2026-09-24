"""Copy cm-chessboard's core ES modules from site/node_modules into site/vendor (MIT, src only).

Only the nine files Chessboard.js imports are vendored: no extensions, and nothing from its assets/
(its pieces/standard.svg and pieces/staunty.svg are share-alike licensed and banned here). The only
change is ASCII punctuation in comments (an ellipsis and em dashes), for the house text rule;
test_vendored_cm_chessboard_matches_npm_except_ascii_punctuation proves nothing else differs.
chess.js is not vendored: its ESM build is 3,368 lines, so `blink site serve` maps it from node_modules.
"""

import shutil
from pathlib import Path

CM_CHESSBOARD_FILES = (
    "Chessboard.js",
    "lib/Svg.js",
    "lib/Utils.js",
    "model/ChessboardState.js",
    "model/Extension.js",
    "model/Position.js",
    "view/ChessboardView.js",
    "view/PositionAnimationsQueue.js",
    "view/VisualMoveInput.js",
)
ASCII = {
    chr(0x2026): "...",
    chr(0x2014): "-",
    chr(0x2013): "-",
    chr(0x201C): '"',
    chr(0x201D): '"',
    chr(0x2018): "'",
    chr(0x2019): "'",
}


def to_ascii_punctuation(text: str) -> str:
    for char, ascii_text in ASCII.items():
        text = text.replace(char, ascii_text)
    return text


def vendor_cm_chessboard(package: Path, out: Path) -> list[Path]:
    """Copy the core modules and the LICENSE; returns the written paths."""
    written = []
    for name in CM_CHESSBOARD_FILES:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        text = to_ascii_punctuation((package / "src" / name).read_text(encoding="utf-8"))
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        written.append(target)
    shutil.copyfile(package / "LICENSE", out / "LICENSE")
    written.append(out / "LICENSE")
    return written
