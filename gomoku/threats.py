"""Find winning points and contiguous opponent threats in Gomoku."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from functools import lru_cache

import numpy as np

from game import DIRECTIONS, EMPTY, Gomoku


Move = tuple[int, int]
Run = tuple[int, tuple[Move, ...]]
FORCING_RUNS = ("open four", "alive three")
WINDOW_WEIGHTS = np.array((0, 1, 24, 600, 20_000, 0))


def nearby_moves(game: Gomoku) -> list[Move]:
    """Find legal moves within two squares of a stone, including diagonals."""
    if game.game_over:
        return []
    rows, columns = np.nonzero(game.board)
    nearby = np.zeros_like(game.board, dtype=bool)
    for row, col in zip(rows.tolist(), columns.tolist()):
        nearby[max(0, row - 2):row + 3, max(0, col - 2):col + 3] = True
    rows, columns = np.nonzero(nearby & (game.board == EMPTY))
    moves = [(col + 1, row + 1) for row, col in zip(rows.tolist(), columns.tolist())]
    center = game.size // 2 + 1
    moves.sort(key=lambda move: abs(move[0] - center) + abs(move[1] - center))
    return moves


def _run_through(
    board: list[list[int]], player: int, x: int, y: int, dx: int, dy: int
) -> Run:
    """Count a run through an actual or hypothetical stone at zero-based x, y."""
    size = len(board)
    length = 1
    open_ends = []
    for sign in (-1, 1):
        step_x, step_y = sign * dx, sign * dy
        end_x, end_y = x + step_x, y + step_y
        while 0 <= end_x < size and 0 <= end_y < size:
            cell = board[end_y][end_x]
            if cell != player:
                if cell == EMPTY:
                    open_ends.append((end_x + 1, end_y + 1))
                break
            length += 1
            end_x += step_x
            end_y += step_y
    return length, tuple(open_ends)


@lru_cache(maxsize=None)
def _five_cell_windows(size: int) -> np.ndarray:
    """Cache flat indexes for every five-cell window on a board of this size."""
    windows = []
    for y in range(size):
        for x in range(size):
            for dx, dy in DIRECTIONS:
                if 0 <= x + 4 * dx < size and 0 <= y + 4 * dy < size:
                    windows.append([(y + i * dy) * size + x + i * dx for i in range(5)])
    return np.asarray(windows, dtype=np.intp)


def _winning_cells(
    windows: np.ndarray, cells: np.ndarray, sums: np.ndarray, player: int
) -> set[int]:
    # A sum of +4 or -4 over five cells is four matching stones and one empty.
    matching = sums == 4 * player
    return set(windows[matching][cells[matching] == EMPTY].tolist())


def _winning_moves(board: np.ndarray, player: int, moves: list[Move]) -> list[Move]:
    if not moves or np.count_nonzero(board == player) < 4:
        return []
    size = len(board)
    windows = _five_cell_windows(size)
    cells = board.ravel()[windows]
    winning = _winning_cells(windows, cells, cells.sum(axis=1), player)
    return [(x, y) for x, y in moves if (y - 1) * size + x - 1 in winning]


def winning_moves(game: Gomoku, player: int, moves: list[Move] | None = None) -> list[Move]:
    """Find every nearby move that wins for player, including moves in a gap."""
    if moves is None:
        moves = nearby_moves(game)
    return _winning_moves(game.board, player, moves)


def search_moves(game: Gomoku, width: int | None) -> list[Move]:
    """Order nearby moves by tactics and optionally keep only the strongest ones."""
    moves = nearby_moves(game)
    if width is None or not moves:
        return moves

    size = game.size
    player = game.current_player
    windows = _five_cell_windows(size)
    cells = game.board.ravel()[windows]
    sums = cells.sum(axis=1)
    wins = _winning_cells(windows, cells, sums, player)
    blocks = _winning_cells(windows, cells, sums, -player)

    # A window with both colors cannot become a five. Among uncontested
    # windows, prioritize near-complete lines, including gapped ones.
    lengths = np.abs(sums)
    weights = np.where(lengths == np.count_nonzero(cells, axis=1), WINDOW_WEIGHTS[lengths], 0)
    scores = np.bincount(windows.ravel(), weights=np.repeat(weights, 5), minlength=size * size)

    def priority(move: Move) -> tuple[bool, bool, float]:
        x, y = move
        index = (y - 1) * size + x - 1
        return index in wins, index in blocks, scores[index]

    # The stable sort keeps the center-first order among equal-scoring moves.
    moves.sort(key=priority, reverse=True)
    return moves[:width]


def _iter_runs(board: list[list[int]], player: int) -> Iterator[Run]:
    size = len(board)
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell != player:
                continue
            for dx, dy in DIRECTIONS:
                previous_x, previous_y = x - dx, y - dy
                if (
                    0 <= previous_x < size
                    and 0 <= previous_y < size
                    and board[previous_y][previous_x] == player
                ):
                    continue
                next_x, next_y = x + dx, y + dy
                if (
                    0 <= next_x < size
                    and 0 <= next_y < size
                    and board[next_y][next_x] == player
                ):
                    yield _run_through(board, player, x, y, dx, dy)


def _threat_name(run: Run) -> str | None:
    length, open_ends = run
    if length == 4 and open_ends:
        return "open four"
    if length == 3:
        if len(open_ends) == 2:
            return "alive three"
        if len(open_ends) == 1:
            return "open three"
    return None


def _best_block(blocks: Counter[Move], size: int) -> Move:
    center = (size + 1) // 2

    def priority(move: Move) -> tuple[int, int, int, int]:
        x, y = move
        distance_squared = (x - center) ** 2 + (y - center) ** 2
        return -blocks[move], distance_squared, y, x

    return min(blocks, key=priority)


def choose_block(game: Gomoku) -> tuple[Move, str] | None:
    """Choose a legal block for the opponent's highest priority threat."""
    if game.game_over:
        return None

    board = game.board.tolist()
    opponent = -game.current_player
    moves = nearby_moves(game)
    immediate_wins = _winning_moves(game.board, opponent, moves)
    if immediate_wins:
        return _best_block(Counter(immediate_wins), game.size), "alive four"

    blocks: dict[str, Counter[Move]] = {threat: Counter() for threat in FORCING_RUNS}
    for run in _iter_runs(board, opponent):
        threat = _threat_name(run)
        if threat is not None and threat in blocks:
            blocks[threat].update(run[1])

    for threat in FORCING_RUNS:
        if blocks[threat]:
            return _best_block(blocks[threat], game.size), threat

    double_threes: Counter[Move] = Counter()
    for x, y in moves:
        # Each axis contributes at most one run through the new stone. Two
        # open endpoints on a single four still count as only one threat.
        threats_created = sum(
            _threat_name(_run_through(board, opponent, x - 1, y - 1, dx, dy)) in FORCING_RUNS
            for dx, dy in DIRECTIONS
        )
        if threats_created >= 2:
            double_threes[(x, y)] = threats_created

    if double_threes:
        return _best_block(double_threes, game.size), "double three"
    return None
