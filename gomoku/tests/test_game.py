"""Behavior checks for the Gomoku rules and reusable game state."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game import BLACK, EMPTY, WHITE, Gomoku, InvalidMove


class GomokuTests(unittest.TestCase):
    def play_alternating(self, game, black_moves, white_moves):
        """Play a legal fixture, asserting it does not finish prematurely."""
        sequence = []
        for index, move in enumerate(black_moves):
            sequence.append(move)
            if index < len(white_moves):
                sequence.append(white_moves[index])
        for index, move in enumerate(sequence):
            self.assertFalse(game.game_over, f"Game ended before move {index + 1}")
            game.play(*move)

    def test_supported_boards_start_empty_with_black_to_move(self):
        for size in (9, 11, 15):
            with self.subTest(size=size):
                game = Gomoku(size)
                self.assertIsInstance(game.board, np.ndarray)
                self.assertEqual(game.board.shape, (size, size))
                self.assertTrue(np.all(game.board == EMPTY))
                self.assertEqual(game.current_player, BLACK)
                self.assertEqual(game.move_count, 0)
                self.assertIsNone(game.last_move)
                self.assertIsNone(game.result)
                self.assertFalse(game.game_over)
                self.assertEqual(len(game.legal_moves()), size * size)

    def test_one_based_xy_coordinates_and_turns(self):
        game = Gomoku(11)
        game.play(11, 2)
        self.assertEqual(game.board[1, 10], BLACK)
        self.assertEqual(game.current_player, WHITE)
        game.play(2, 11)
        self.assertEqual(game.board[10, 1], WHITE)
        self.assertEqual(game.current_player, BLACK)
        self.assertEqual(game.move_count, 2)
        self.assertEqual(game.last_move, (2, 11))
        self.assertNotIn((11, 2), game.legal_moves())
        self.assertNotIn((2, 11), game.legal_moves())

    def test_rejected_moves_leave_every_part_of_state_unchanged(self):
        invalid_moves = [(1, 1), (0, 2), (-1, 3), (10, 1), (1, 10),
                         (1.5, 2), (2, 1.0), (True, 2), (2, False), ("2", 3)]
        for move in invalid_moves:
            with self.subTest(move=move):
                game = Gomoku(9)
                game.play(1, 1)
                board_before = game.board.copy()
                with self.assertRaises(InvalidMove):
                    game.play(*move)
                np.testing.assert_array_equal(game.board, board_before)
                self.assertEqual(game.current_player, WHITE)
                self.assertEqual(game.move_count, 1)
                self.assertEqual(game.last_move, (1, 1))
                self.assertIsNone(game.winner)
                self.assertIsNone(game.result)
                self.assertFalse(game.is_draw)
                game.play(2, 1)
                self.assertEqual(game.board[0, 1], WHITE)

    def test_black_wins_in_all_four_directions_at_board_edges(self):
        fixtures = [
            ([(5, 9), (6, 9), (7, 9), (8, 9), (9, 9)],
             [(1, 1), (3, 1), (5, 1), (7, 1)]),
            ([(9, 1), (9, 2), (9, 3), (9, 4), (9, 5)],
             [(1, 9), (3, 9), (5, 9), (7, 9)]),
            ([(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
             [(1, 1), (3, 1), (5, 1), (7, 1)]),
            ([(5, 5), (6, 4), (7, 3), (8, 2), (9, 1)],
             [(1, 9), (3, 9), (5, 9), (7, 9)]),
        ]
        for black_moves, white_moves in fixtures:
            with self.subTest(line=black_moves):
                game = Gomoku(9)
                self.play_alternating(game, black_moves, white_moves)
                self.assertEqual(game.winner, BLACK)
                self.assertEqual(game.result, 1)
                self.assertTrue(game.game_over)
                self.assertFalse(game.is_draw)
                self.assertEqual(game.current_player, BLACK)
                self.assertEqual(game.legal_moves(), [])

    def test_white_can_win_and_has_negative_terminal_value(self):
        game = Gomoku(9)
        self.play_alternating(
            game,
            [(1, 1), (3, 1), (5, 1), (7, 1), (9, 1)],
            [(5, 5), (5, 6), (5, 7), (5, 8), (5, 9)],
        )
        self.assertEqual(game.winner, WHITE)
        self.assertEqual(game.result, -1)
        self.assertEqual(game.current_player, WHITE)

    def test_joining_two_segments_into_six_is_a_win(self):
        game = Gomoku(9)
        self.play_alternating(
            game,
            [(2, 5), (3, 5), (4, 5), (6, 5), (7, 5), (5, 5)],
            [(1, 1), (3, 1), (5, 1), (7, 1), (9, 1)],
        )
        self.assertEqual(game.winner, BLACK)
        self.assertEqual(game.move_count, 11)

    def test_a_gap_or_row_boundary_does_not_create_a_line(self):
        for moves in (
            [(1, 5), (2, 5), (3, 5), (5, 5), (6, 5)],
            [(7, 4), (8, 4), (9, 4), (1, 5), (2, 5)],
        ):
            with self.subTest(moves=moves):
                game = Gomoku(9)
                self.play_alternating(
                    game, moves, [(1, 1), (3, 1), (5, 1), (7, 1)]
                )
                self.assertIsNone(game.winner)
                self.assertFalse(game.game_over)

    def test_game_rejects_moves_after_a_win(self):
        game = Gomoku(9)
        self.play_alternating(
            game,
            [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
            [(1, 3), (3, 3), (5, 3), (7, 3)],
        )
        board_before = game.board.copy()
        with self.assertRaises(InvalidMove):
            game.play(9, 9)
        np.testing.assert_array_equal(game.board, board_before)
        self.assertEqual(game.move_count, 9)
        self.assertEqual(game.winner, BLACK)

    def test_full_board_without_five_in_a_row_is_a_draw(self):
        # This stripe pattern has horizontal runs of two, vertical runs of one,
        # and diagonal runs of at most two. It contains 41 black and 40 white
        # cells, so the whole board can be reached through alternating moves.
        black_moves = []
        white_moves = []
        for y in range(9):
            for x in range(9):
                target = black_moves if (x + 2 * y) % 4 < 2 else white_moves
                target.append((x + 1, y + 1))
        game = Gomoku(9)
        self.play_alternating(game, black_moves, white_moves)
        self.assertTrue(game.is_draw)
        self.assertTrue(game.game_over)
        self.assertIsNone(game.winner)
        self.assertEqual(game.result, 0)
        self.assertEqual(game.move_count, 81)
        self.assertEqual(game.legal_moves(), [])
        with self.assertRaises(InvalidMove):
            game.play(1, 1)

    def test_copy_can_be_searched_without_mutating_the_real_game(self):
        game = Gomoku(15)
        game.play(8, 8)
        branch = game.copy()
        self.assertEqual(branch.last_move, (8, 8))
        self.assertEqual(branch.current_player, WHITE)
        branch.play(8, 9)
        self.assertEqual(branch.board[8, 7], WHITE)
        self.assertEqual(game.board[8, 7], EMPTY)
        self.assertEqual(game.current_player, WHITE)
        self.assertEqual(game.move_count, 1)
        self.assertEqual(branch.move_count, 2)

    def test_render_labels_large_board_and_shows_both_stones(self):
        game = Gomoku(15)
        empty_display = game.render()
        self.assertIn("15", empty_display)
        self.assertIn("+", empty_display)
        game.play(15, 1)
        game.play(1, 15)
        display = game.render()
        self.assertIn("X", display)
        self.assertIn("O", display)
        self.assertIn("15", display)


if __name__ == "__main__":
    unittest.main()
