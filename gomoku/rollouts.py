"""Estimate short-horizon losses from sampled tactical continuations."""

from __future__ import annotations

from random import Random

from game import Gomoku
from threats import nearby_moves, winning_moves


def count_losses(
    game: Gomoku, player: int, remaining_moves: int, samples: int, seed: int
) -> int:
    """Count opponent wins by the horizon from a position after a candidate move.

    A draw or a game still in progress at the horizon contributes no loss.
    """
    rng = Random(seed)
    losses = 0
    for _ in range(samples):
        position = game.copy()
        for _ in range(remaining_moves):
            if position.game_over:
                break
            moves = nearby_moves(position)
            side = position.current_player
            choices = (
                winning_moves(position, side, moves)
                or winning_moves(position, -side, moves)
                or moves
            )
            position.play(*rng.choice(choices))
        if position.winner == -player:
            losses += 1
    return losses
