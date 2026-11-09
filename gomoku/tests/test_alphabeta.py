"""Threat definitions and search behavior for the alpha-beta player."""

import sys
import unittest
from math import inf
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphabeta import AlphaBetaAgent, _minimax
from game import BLACK, WHITE, Gomoku
from rollouts import count_losses
from threats import choose_block


FILLERS = [(x, y) for y in (1, 3, 5, 7, 9, 11) for x in (9, 11)]


def black_to_move(white_moves, black_moves=None):
    """Reach a Black turn through legal moves, with isolated Black fillers by default."""
    game = Gomoku(11)
    if black_moves is None:
        black_moves = FILLERS[:len(white_moves)]
    for black_move, white_move in zip(black_moves, white_moves):
        game.play(*black_move)
        game.play(*white_move)
    assert game.current_player == BLACK and not game.game_over
    return game


class ThreatTests(unittest.TestCase):
    def test_open_four_with_one_end_at_the_edge_in_every_direction(self):
        cases = [
            ([(1, 4), (2, 4), (3, 4), (4, 4)], (5, 4)),
            ([(4, 1), (4, 2), (4, 3), (4, 4)], (4, 5)),
            ([(1, 1), (2, 2), (3, 3), (4, 4)], (5, 5)),
            ([(1, 11), (2, 10), (3, 9), (4, 8)], (5, 7)),
        ]
        for stones, block in cases:
            with self.subTest(stones=stones):
                self.assertEqual(choose_block(black_to_move(stones)), (block, "alive four"))

    def test_open_four_with_one_end_occupied_by_defender(self):
        game = black_to_move(
            [(2, 4), (3, 4), (4, 4), (5, 4)],
            [(1, 4), *FILLERS[:3]],
        )
        self.assertEqual(choose_block(game), ((6, 4), "alive four"))

    def test_alive_three_requires_two_empty_ends(self):
        alive_three = black_to_move([(3, 5), (4, 5), (5, 5)])
        move, reason = choose_block(alive_three)
        self.assertIn(move, {(2, 5), (6, 5)})
        self.assertEqual(reason, "alive three")

        one_end_at_edge = black_to_move([(1, 5), (2, 5), (3, 5)])
        one_end_occupied = black_to_move(
            [(3, 5), (4, 5), (5, 5)], [(2, 5), *FILLERS[:2]]
        )
        self.assertIsNone(choose_block(one_end_at_edge))
        self.assertIsNone(choose_block(one_end_occupied))

    def test_all_three_double_three_combinations(self):
        # (5, 5) joins the separate horizontal and vertical segments.
        horizontal_three = [(4, 5), (6, 5)]
        vertical_three = [(5, 4), (5, 6)]
        cases = [
            ("three and three", horizontal_three + vertical_three),
            ("three and four", horizontal_three + vertical_three + [(3, 5)]),
            ("four and four", horizontal_three + vertical_three + [(3, 5), (5, 3)]),
        ]
        for description, stones in cases:
            with self.subTest(combination=description):
                game = black_to_move(stones)
                board_before = game.board.copy()
                self.assertEqual(choose_block(game), ((5, 5), "double three"))
                np.testing.assert_array_equal(game.board, board_before)

    def test_two_ends_of_one_hypothetical_four_are_one_threat(self):
        # Playing (5, 5) creates a four with two ends, but only one run.
        game = black_to_move([(3, 5), (4, 5), (6, 5)])
        self.assertIsNone(choose_block(game))

    def test_open_three_does_not_count_toward_double_three(self):
        # (5, 5) would make an alive three vertically and a three blocked at (3, 5).
        game = black_to_move(
            [(4, 5), (6, 5), (5, 4), (5, 6)], [(3, 5), *FILLERS[:3]]
        )
        self.assertIsNone(choose_block(game))

    def test_own_open_four_is_not_a_defensive_threat(self):
        game = black_to_move(FILLERS[:4], [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertIsNone(choose_block(game))

    def test_alive_three_takes_precedence_over_double_three(self):
        game = black_to_move([
            (2, 8), (3, 8), (4, 8),  # Existing alive three.
            (6, 4), (8, 4), (7, 3), (7, 5),  # Double three at (7, 4).
        ])
        move, reason = choose_block(game)
        self.assertIn(move, {(1, 8), (5, 8)})
        self.assertEqual(reason, "alive three")


class AlphaBetaTests(unittest.TestCase):
    def test_search_counts_a_win_on_the_tenth_move(self):
        game = Gomoku(9)
        game.play(5, 5)
        game.play(3, 3)  # White's root candidate counts as the first move.
        continuation = (
            (2, 4), (5, 3), (3, 4), (7, 3), (4, 4),
            (7, 5), (5, 4), (7, 7), (6, 4),
        )

        def only_move(position, width):
            return [continuation[position.move_count - game.move_count]]

        with patch("alphabeta.search_moves", side_effect=only_move):
            # The final Black stone completes five. One fewer turn leaves
            # the same line unfinished and therefore neutral for minimax.
            self.assertEqual(_minimax(game, 8, -inf, inf, WHITE, None), 0)
            self.assertEqual(_minimax(game, 9, -inf, inf, WHITE, None), -1)

    def test_agent_prefers_the_lower_sampled_loss_rate_on_a_tie(self):
        game = Gomoku(9)
        game.play(5, 5)
        riskier, safer = (5, 4), (6, 5)

        def sampled_losses(position, player, remaining_moves, samples, seed):
            self.assertEqual(player, WHITE)
            self.assertEqual(position.move_count, game.move_count + 1)
            self.assertEqual(remaining_moves, 9)
            return samples if position.last_move == riskier else 0

        def unresolved(position, depth, alpha, beta, player, width):
            self.assertEqual(depth, 9)
            return 0

        with (
            patch("alphabeta.nearby_moves", return_value=[riskier, safer]),
            patch("alphabeta.count_losses", side_effect=sampled_losses) as estimate,
            patch("alphabeta._minimax", side_effect=unresolved),
        ):
            self.assertEqual(AlphaBetaAgent().choose_move(game), safer)
        self.assertEqual(estimate.call_count, 2)

    def test_sampled_losses_count_only_wins_within_the_horizon(self):
        game = black_to_move(FILLERS[:4], [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertEqual(count_losses(game, WHITE, 0, samples=4, seed=0), 0)
        self.assertEqual(count_losses(game, WHITE, 1, samples=4, seed=0), 4)
        self.assertEqual(game.move_count, 8)
        self.assertFalse(game.game_over)

    def test_agent_blocks_immediate_win_before_other_threats(self):
        game = black_to_move([
            (1, 3), (2, 3), (3, 3), (4, 3),  # Open four at the edge.
            (2, 8), (3, 8), (4, 8),  # Alive three.
            (6, 5), (8, 5), (7, 4), (7, 6),  # Double three at (7, 5).
        ])
        agent = AlphaBetaAgent()
        self.assertEqual(agent.choose_move(game), (5, 3))
        self.assertEqual(agent.last_reason, "block alive four")

    def test_agent_blocks_gapped_five_or_six_ahead_of_alive_three(self):
        gapped_four = [(1, 4), (2, 4), (4, 4), (5, 4)]
        alive_three = [(5, 8), (6, 8), (7, 8)]
        for extra in ([], [(6, 4)]):
            with self.subTest(overline=bool(extra)):
                game = black_to_move(gapped_four + extra + alive_three)
                agent = AlphaBetaAgent()
                self.assertEqual(agent.choose_move(game), (3, 4))
                self.assertEqual(agent.last_reason, "block alive four")

    def test_search_finds_the_only_winning_move_five_plies_ahead(self):
        # This endgame has nine empty cells and no direct block or win. Only
        # (5, 9) leads to a forced Black win, on the fifth ply. Every other
        # first move lets White force a draw or win.
        rows = (
            "OXOXXXOOX",
            "OXXXOOXOX",
            "XXOOXXOOX",
            "OOXOOOXXO",
            "XOOOXXOOX",
            "XOO...XXO",
            "XXO.X.OOX",
            "OOXO.O.XO",
            "OXOX.X.XX",
        )
        stones = {symbol: [] for symbol in "XO"}
        for y, row in enumerate(rows, 1):
            for x, symbol in enumerate(row, 1):
                if symbol in stones:
                    stones[symbol].append((x, y))
        game = Gomoku(9)
        for black_move, white_move in zip(stones["X"], stones["O"]):
            game.play(*black_move)
            game.play(*white_move)

        board_before = game.board.copy()
        state_before = (game.current_player, game.last_move, game.move_count, game.result)
        agent = AlphaBetaAgent()

        move = agent.choose_move(game)

        self.assertEqual(move, (5, 9))
        self.assertEqual(agent.last_reason, "alpha-beta search")
        np.testing.assert_array_equal(game.board, board_before)
        self.assertEqual(
            (game.current_player, game.last_move, game.move_count, game.result), state_before
        )

    def test_agent_plays_the_center_of_an_empty_board(self):
        for size in (9, 11, 15):
            with self.subTest(size=size):
                game = Gomoku(size)
                center = size // 2 + 1
                self.assertEqual(AlphaBetaAgent().choose_move(game), (center, center))
                self.assertEqual(game.move_count, 0)


if __name__ == "__main__":
    unittest.main()
