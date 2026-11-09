"""Input parsing and a full keyboard-driven Gomoku game."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import play
from game import WHITE


class InputParsingTests(unittest.TestCase):
    def test_supported_coordinate_formats(self):
        for text, expected in [
            ("3 5", (3, 5)),
            ("  11   15  ", (11, 15)),
            ("(1, 9)", (1, 9)),
            ("11,15", (11, 15)),
            ("  ( 3 , 4 )  ", (3, 4)),
            ("-1, 0", (-1, 0)),
        ]:
            with self.subTest(text=text):
                self.assertEqual(play.parse_move(text), expected)

    def test_malformed_coordinates_are_rejected(self):
        for text in ("", "one,two", "1", "1 2 3", "1,2,3", "1.5,2", "(1,2", "1,2)",
                     "[1,2]", "(1,2) trailing", "__import__('os')"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    play.parse_move(text)

    def test_board_size_formats(self):
        for text, expected in [("9", 9), ("11", 11), ("15", 15),
                               ("9x9", 9), ("11 * 11", 11), ("15 x 15", 15)]:
            with self.subTest(text=text):
                self.assertEqual(play.parse_size(text), expected)

    def test_invalid_board_sizes_are_rejected(self):
        for text in ("", "5", "10", "9x11", "15*9", "9.0", "abc"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    play.parse_size(text)


class InteractiveGameTests(unittest.TestCase):
    def run_game(self, inputs, agent_side=None):
        prompts = []
        answers = iter(inputs)

        def fake_input(prompt=""):
            prompts.append(prompt)
            try:
                return next(answers)
            except StopIteration:
                self.fail("The game requested more input than the fixture provides")

        output = io.StringIO()
        with patch("builtins.input", side_effect=fake_input), contextlib.redirect_stdout(output):
            exit_code = play.main(agent_side=agent_side)
        return exit_code, prompts, output.getvalue()

    def test_full_game_retries_invalid_moves_and_stops_at_black_win(self):
        exit_code, prompts, output = self.run_game([
            "10", "9",
            "not a coordinate", "1 1",
            "1 1", "10 1", "1 3",
            "2 1", "3 3",
            "3 1", "5 3",
            "4 1", "7 3",
            "5 1",
        ])
        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(output.lower().count("invalid"), 4)
        self.assertRegex(output.lower(), r"black[^\n]*win")
        self.assertIn("X", output)
        self.assertIn("O", output)
        move_prompts = [prompt.lower() for prompt in prompts
                        if "black" in prompt.lower() or "white" in prompt.lower()]
        self.assertEqual(len(move_prompts), 12)
        self.assertIn("black", move_prompts[0])
        self.assertIn("black", move_prompts[1])
        self.assertTrue(all("white" in prompt for prompt in move_prompts[2:5]))
        self.assertIn("black", move_prompts[-1])

    def test_agent_move_uses_no_keyboard_input(self):
        with patch("play.AlphaBetaAgent") as agent_class:
            agent = agent_class.return_value
            agent.choose_move.return_value = (5, 5)
            agent.last_reason = "alpha-beta search"
            exit_code, prompts, output = self.run_game(["9", "1 1", "q"], agent_side=WHITE)

        self.assertEqual(exit_code, 0)
        agent.choose_move.assert_called_once()
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all("Black (X)" in prompt for prompt in prompts[1:]))
        self.assertIn("White (O, AI) played (5, 5)", output)
        self.assertRegex(output, r"(?m)^ 5 \|(?:\s+\+){4}\s+O")

    def test_quit_at_size_prompt(self):
        exit_code, _, _ = self.run_game(["q"])
        self.assertEqual(exit_code, 0)

    def test_quit_during_game(self):
        exit_code, _, _ = self.run_game(["11", "(1, 1)", "quit"])
        self.assertEqual(exit_code, 0)

    def test_end_of_input_exits_without_traceback(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(output):
            exit_code = play.main()
        self.assertEqual(exit_code, 0)
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
