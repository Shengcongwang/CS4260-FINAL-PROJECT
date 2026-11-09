"""A defensive Gomoku agent with a ten-move search horizon."""

from __future__ import annotations

from math import inf

from game import Gomoku
from rollouts import count_losses
from threats import Move, choose_block, nearby_moves, search_moves, winning_moves


SEARCH_DEPTH = 10
SEARCH_WIDTH = 3
ROLLOUT_SAMPLES = 16
ROLLOUT_SEED = 0


def _minimax(
    game: Gomoku, depth: int, alpha: float, beta: float, player: int, width: int | None
) -> float:
    result = game.result
    if result is not None:
        return result * player
    if depth == 0:
        # The position is still unresolved at the horizon.
        return 0

    maximizing = game.current_player == player
    best = -inf if maximizing else inf
    for move in search_moves(game, width):
        child = game.copy()
        child.play(*move)
        score = _minimax(child, depth - 1, alpha, beta, player, width)
        if maximizing:
            best = max(best, score)
            alpha = max(alpha, best)
        else:
            best = min(best, score)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best


class AlphaBetaAgent:
    """Block threats before searching. Moves use the game's one-based coordinates."""

    def __init__(self) -> None:
        self.last_reason = ""

    def choose_move(self, game: Gomoku) -> Move:
        """Choose a legal move without modifying the game."""
        if game.game_over:
            raise ValueError("Cannot choose a move after the game ends.")
        if game.move_count == 0:
            center = game.size // 2 + 1
            self.last_reason = "opening"
            return center, center

        block = choose_block(game)
        if block is not None:
            move, threat = block
            self.last_reason = f"block {threat}"
            return move

        player = game.current_player
        # An immediate win is the highest possible search score.
        wins = winning_moves(game, player)
        if wins:
            self.last_reason = "immediate win"
            return wins[0]

        # With at most ten empty squares every path can reach a terminal state.
        # Search that small endgame with the full set of moves and skip sampling.
        sample = game.size ** 2 - game.move_count > SEARCH_DEPTH
        width = SEARCH_WIDTH if sample else None
        candidates = []
        for move in nearby_moves(game):
            child = game.copy()
            child.play(*move)
            losses = 0
            if sample:
                losses = count_losses(
                    child, player, SEARCH_DEPTH - 1, ROLLOUT_SAMPLES, ROLLOUT_SEED
                )
            candidates.append((losses, move, child))

        # For equal minimax outcomes, retain the first move, which has the lower
        # sampled loss rate. Stable sorting also keeps the original tie order.
        candidates.sort(key=lambda candidate: candidate[0])
        best_move = candidates[0][1]
        best_losses = candidates[0][0]
        best_score = -inf
        for losses, move, child in candidates:
            score = _minimax(child, SEARCH_DEPTH - 1, best_score, 1, player, width)
            if score > best_score:
                best_move, best_score = move, score
                best_losses = losses
            if best_score == 1:
                break

        self.last_reason = "alpha-beta search"
        if sample and best_score == 0:
            self.last_reason += f", estimated loss {best_losses / ROLLOUT_SAMPLES:.1%}"
        return best_move
