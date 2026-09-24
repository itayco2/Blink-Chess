"""The cburnett piece sprite for cm-chessboard, built from the 12 Wikimedia Commons SVGs.

Each file Chess_<piece><l|d>t45.svg is a 45x45 drawing by Cburnett (2006). cm-chessboard draws a piece
with <use href="#wk">, so every drawing becomes <g id="wk"> in one sprite, and the page sets
pieces.tileSize = 45. Inner ids are dropped so no two pieces collide. The licence election lives in
site/assets/pieces/LICENSE-pieces.txt.
"""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
TILE_SIZE = 45
COLORS = {"w": "l", "b": "d"}
PIECES = "kqrbnp"
ATTRIBUTION = (
    " Chess pieces by Cburnett (Wikimedia Commons, 2006), used under the BSD 3-Clause License as elected "
    "in LICENSE-pieces.txt. Source: https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces "
)


def source_names() -> dict[str, str]:
    """cm-chessboard piece id -> Commons file name, e.g. wk -> Chess_klt45.svg."""
    return {f"{c}{p}": f"Chess_{p}{shade}t{TILE_SIZE}.svg" for c, shade in COLORS.items() for p in PIECES}


def _piece_group(piece_id: str, svg_text: str) -> ET.Element:
    root = ET.fromstring(svg_text)
    group = ET.Element(f"{{{SVG_NS}}}g", {"id": piece_id})
    for child in list(root):
        for element in child.iter():
            element.attrib.pop("id", None)
        group.append(child)
    return group


def build_sprite(sources: Mapping[str, str]) -> str:
    """One SVG holding <g id="wk">..<g id="bp">, from piece id -> SVG text."""
    missing = sorted(set(source_names()) - set(sources))
    if missing:
        raise ValueError(f"missing pieces: {missing}")
    ET.register_namespace("", SVG_NS)
    sprite = ET.Element(
        f"{{{SVG_NS}}}svg",
        {"width": str(TILE_SIZE), "height": str(TILE_SIZE), "viewBox": f"0 0 {TILE_SIZE} {TILE_SIZE}"},
    )
    sprite.append(ET.Comment(ATTRIBUTION))
    for piece_id in source_names():
        sprite.append(_piece_group(piece_id, sources[piece_id]))
    ET.indent(sprite)
    return ET.tostring(sprite, encoding="unicode") + "\n"


def write_sprite(src_dir: Path, out: Path) -> Path:
    names = source_names()
    sources = {piece_id: (src_dir / name).read_text(encoding="utf-8") for piece_id, name in names.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_sprite(sources))
    return out
