"""`blink site smoke --url URL`: drive the page in the installed Edge and check what a visitor sees.

Playwright launches channel "msedge" (the system Edge), so no browser is ever downloaded. The smoke
plays random legal user moves by clicking squares, waits for each of Blink's replies, replays every
game in python-chess to prove each reply legal, counts the arrows on the board, and records every
console error, page error and failed request. `failures()` turns the report into a verdict.
A reply shows one arrow per move of the network's top 3 (fewer when fewer moves are legal), and none
after a mate in one, which rule R2 plays without a network call.
"""

import random
import statistics
import time
from dataclasses import asdict, dataclass

import chess

ARROWS_EXPECTED = 3
MAX_USER_MOVES_FACTOR = 3  # a game that ends early starts a new one; give up after 3x the target
WAIT_READY = "() => window.__blink && window.__blink.state().ready"
WAIT_USER_TURN = (
    "() => { const s = window.__blink.state(); "
    "return s.gameOver || (!s.thinking && s.turn === s.userColor); }"
)
WAIT_REPLY = "n => { const s = window.__blink.state(); return s.replies >= n || s.gameOver; }"
WAIT_NEW_GAME = "() => { const s = window.__blink.state(); return s.history.length === 0 && !s.thinking; }"


@dataclass(frozen=True)
class SmokeReport:
    url: str
    loaded: bool
    user_moves: int
    legal_replies: int
    illegal: tuple[str, ...]
    arrows: int
    min_arrows: int
    console_errors: tuple[str, ...]
    timings_ms: tuple[float, ...]
    games: int
    load_seconds: float
    problems: tuple[str, ...] = ()
    expected_arrows: int = ARROWS_EXPECTED
    last_rule: str | None = None

    def to_dict(self) -> dict:
        timings = list(self.timings_ms)
        summary = {
            "ms_per_move_median": round(statistics.median(timings), 2) if timings else None,
            "ms_per_move_max": round(max(timings), 2) if timings else None,
            "moves_timed": len(timings),
        }
        return {**asdict(self), **summary}


def replay(start_fen: str, history: list[str], user_color: str) -> tuple[int, list[str]]:
    """(Blink's legal replies, the first illegal move if any), replaying `history` from `start_fen`."""
    board = chess.Board(start_fen)
    replies = 0
    for ply, uci in enumerate(history):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return replies, [f"ply {ply}: {uci} is illegal in {board.fen()}"]
        mover = "w" if board.turn == chess.WHITE else "b"
        replies += mover != user_color
        board.push(move)
    return replies, []


def expected_arrows(start_fen: str, history: list[str], last_rule: str | None) -> int:
    """Arrows after the last ply (Blink's reply): min(3, its legal moves), or 0 when R2 skipped the look."""
    if not history or last_rule == "R2":
        return 0
    board = chess.Board(start_fen)
    for uci in history[:-1]:
        board.push_uci(uci)
    return min(ARROWS_EXPECTED, board.legal_moves.count())


def failures(report: SmokeReport, moves: int) -> list[str]:
    out = list(report.problems)
    if not report.loaded:
        out.append("the page never became ready (model or vocab failed to load)")
    if report.legal_replies < moves:
        out.append(f"{report.legal_replies} legal replies, expected at least {moves}")
    out.extend(f"illegal move: {text}" for text in report.illegal)
    if report.arrows != report.expected_arrows:
        out.append(f"{report.arrows} arrows after the last reply, expected {report.expected_arrows}")
    if report.console_errors:
        count = len(report.console_errors)
        out.append(f"{count} console error{'s' if count != 1 else ''}: {report.console_errors[0]}")
    return out


def _click_square(page, square: str) -> None:
    box = page.locator(f'#board rect.square[data-square="{square}"]').bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def _record_game(page, games: list) -> None:
    state = page.evaluate("window.__blink.state()")
    games.append((state["startFen"], state["history"], state["userColor"]))


@dataclass
class _Progress:
    games: list
    user_moves: int = 0
    min_arrows: int = ARROWS_EXPECTED


def _play_one(page, rng: random.Random, progress: _Progress, timeout_ms: float) -> None:
    """One user move by clicking, then Blink's reply; a finished game is recorded and restarted."""
    page.wait_for_function(WAIT_USER_TURN, timeout=timeout_ms)
    if page.evaluate("window.__blink.state().gameOver"):
        _record_game(page, progress.games)
        page.click("#new-game")
        page.wait_for_function(WAIT_NEW_GAME, timeout=timeout_ms)
        return
    replies = page.evaluate("window.__blink.state().replies")
    move = rng.choice(page.evaluate("window.__blink.legalMoves()"))
    _click_square(page, move["from"])
    _click_square(page, move["to"])
    progress.user_moves += 1
    page.wait_for_function(WAIT_REPLY, arg=replies + 1, timeout=timeout_ms)
    state = page.evaluate("window.__blink.state()")
    if state["replies"] > replies and state["lastRule"] != "R2":
        progress.min_arrows = min(progress.min_arrows, state["arrows"])


def _play(page, moves: int, seed: int, timeout_ms: float, timeout_error: type) -> tuple[_Progress, list]:
    """Click random legal user moves until Blink has replied `moves` times (or a wait times out)."""
    rng = random.Random(seed)
    progress = _Progress(games=[])
    problems = []
    while page.evaluate("window.__blink.state().replies") < moves:
        if progress.user_moves >= moves * MAX_USER_MOVES_FACTOR:
            problems.append(f"gave up after {progress.user_moves} user moves")
            break
        try:
            _play_one(page, rng, progress, timeout_ms)
        except timeout_error:
            problems.append(f"timed out after {progress.user_moves} user moves waiting for the page")
            break
    _record_game(page, progress.games)
    return progress, problems


def _listen(page, errors: list) -> None:
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(f"page error: {exc}"))
    page.on("requestfailed", lambda request: errors.append(f"request failed: {request.url}"))


def run(url: str, moves: int = 10, seed: int = 0, timeout_s: float = 60.0) -> SmokeReport:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    timeout_ms = timeout_s * 1000
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            _listen(page, errors)
            started = time.perf_counter()
            page.goto(url)
            try:
                page.wait_for_function(WAIT_READY, timeout=timeout_ms)
            except PlaywrightTimeout:
                return SmokeReport(url, False, 0, 0, (), 0, 0, tuple(errors), (), 0, timeout_s)
            load_seconds = time.perf_counter() - started
            progress, problems = _play(page, moves, seed, timeout_ms, PlaywrightTimeout)
            state = page.evaluate("window.__blink.state()")
        finally:
            browser.close()
    counted = [replay(start, history, color) for start, history, color in progress.games]
    return SmokeReport(
        url=url,
        loaded=True,
        user_moves=progress.user_moves,
        legal_replies=sum(replies for replies, _ in counted),
        illegal=tuple(text for _, bad in counted for text in bad),
        arrows=state["arrows"],
        min_arrows=progress.min_arrows,
        console_errors=tuple(errors),
        timings_ms=tuple(round(ms, 2) for ms in state["timings"]),
        games=len(progress.games),
        load_seconds=round(load_seconds, 2),
        problems=tuple(problems),
        expected_arrows=expected_arrows(state["startFen"], state["history"], state["lastRule"]),
        last_rule=state["lastRule"],
    )
