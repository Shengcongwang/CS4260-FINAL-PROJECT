"""Gomoku rules and board state, independent of terminal input or AI agents."""

from __future__ import annotations

from numbers import Integral

import numpy as np


EMPTY = 0
BLACK = 1
WHITE = -1
SUPPORTED_SIZES = (9, 11, 15)
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
PLAYER_NAMES = {BLACK: "Black", WHITE: "White"}
SYMBOLS = {EMPTY: "+", BLACK: "X", WHITE: "O"}


class InvalidMove(ValueError):
    """A move cannot be played; the game state has not changed."""


class Gomoku:
    """Freestyle Gomoku: five or more consecutive stones wins.

    All move coordinates are ONE-BASED (x, y): x is the column from the
    left, and y is the row from the top. The NumPy board uses [y - 1, x - 1].
    Black moves first. Use play() to change the board so that turn and result
    tracking stay consistent. There is no input or printing in this class.
    """

    def __init__(self, size: int = 9) -> None:
        if isinstance(size, bool) or not isinstance(size, Integral) or size not in SUPPORTED_SIZES:
            raise ValueError("Board size must be 9, 11, or 15.")
        self.size = int(size)
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = BLACK
        self.winner: int | None = None
        self.is_draw = False
        self.move_count = 0
        self.last_move: tuple[int, int] | None = None

    @property
    def game_over(self) -> bool:
        return self.winner is not None or self.is_draw

    @property
    def result(self) -> int | None:
        """Black's terminal utility: +1 win, -1 loss, 0 draw; None if ongoing."""
        if self.winner is not None:
            return self.winner
        return 0 if self.is_draw else None

    def play(self, x: int, y: int) -> None:
        """Place the current player's stone, or raise InvalidMove atomically.

        The turn advances only after a valid, nonterminal move. At the end
        of a game current_player remains the player who made the final move.
        """
        if self.game_over:
            raise InvalidMove("The game is already over.")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (x, y)):
            raise InvalidMove("Coordinates must be whole numbers.")
        if not (1 <= x <= self.size and 1 <= y <= self.size):
            raise InvalidMove(f"Both coordinates must be between 1 and {self.size}.")
        x, y = int(x), int(y)
        if self.board[y - 1, x - 1] != EMPTY:
            raise InvalidMove(f"Spot ({x}, {y}) is already occupied.")

        self.board[y - 1, x - 1] = self.current_player
        self.move_count += 1
        self.last_move = (x, y)

        if self._has_winning_line(x, y):
            self.winner = self.current_player
        elif self.move_count == self.size * self.size:
            self.is_draw = True
        else:
            self.current_player = -self.current_player

    def _has_winning_line(self, x: int, y: int) -> bool:
        # Only lines through the new stone can have become winning lines.
        for dx, dy in DIRECTIONS:
            count = 1
            for direction in (-1, 1):
                next_x = x + direction * dx
                next_y = y + direction * dy
                while 1 <= next_x <= self.size and 1 <= next_y <= self.size:
                    if self.board[next_y - 1, next_x - 1] != self.current_player:
                        break
                    count += 1
                    next_x += direction * dx
                    next_y += direction * dy
            if count >= 5:
                return True
        return False

    def legal_moves(self) -> list[tuple[int, int]]:
        """Return every empty spot in row order, or [] when the game is over.

        Search-specific restrictions (such as proximity to existing stones)
        belong in the agents. They are not restrictions on legal human moves.
        """
        if self.game_over:
            return []
        return [(int(x) + 1, int(y) + 1) for y, x in np.argwhere(self.board == EMPTY)]

    def copy(self) -> Gomoku:
        """Return independent state for an agent to explore hypothetical moves."""
        duplicate = Gomoku(self.size)
        duplicate.board = self.board.copy()
        duplicate.current_player = self.current_player
        duplicate.winner = self.winner
        duplicate.is_draw = self.is_draw
        duplicate.move_count = self.move_count
        duplicate.last_move = self.last_move
        return duplicate

    def render(self) -> str:
        """Return an ASCII board with numbered x and y axes."""
        lines = ["     x ->", "     " + " ".join(f"{x:2d}" for x in range(1, self.size + 1)), " y"]
        for y, row in enumerate(self.board, start=1):
            lines.append(f"{y:2d} | " + " ".join(f"{SYMBOLS[int(stone)]:>2}" for stone in row))
        return "\n".join(lines)
