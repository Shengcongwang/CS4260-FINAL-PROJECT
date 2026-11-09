"""Play Gomoku against another person or the alpha-beta agent."""

from __future__ import annotations

import argparse
import re

from alphabeta import AlphaBetaAgent
from game import BLACK, EMPTY, WHITE, Gomoku, PLAYER_NAMES, SYMBOLS


QUIT_COMMANDS = {"q", "quit", "exit"}


def parse_size(text: str) -> int:
    """Accept 9, 11, 15, or square dimensions such as 9x9 or 15*15."""
    match = re.fullmatch(r"\s*(9|11|15)(?:\s*[xX*\u00d7]\s*(9|11|15))?\s*", text)
    if match is None or (match[2] is not None and match[1] != match[2]):
        raise ValueError("Choose 9, 11, or 15 (for a 9x9, 11x11, or 15x15 board).")
    return int(match[1])


def parse_move(text: str) -> tuple[int, int]:
    """Accept 'x y' and comma-separated coordinates."""
    text = text.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    match = re.fullmatch(r"\s*([+-]?[0-9]+)(?:\s*,\s*|\s+)([+-]?[0-9]+)\s*", text)
    if match is None:
        raise ValueError("Enter two whole numbers separated by a space, e.g. 3 5.")
    return int(match[1]), int(match[2])


def _read_input(prompt: str) -> str:
    text = input(prompt).strip()
    if text.lower() in QUIT_COMMANDS:
        raise EOFError
    return text


def main(agent_side: int | None = None) -> int:
    if agent_side is None:
        print("Gomoku - two players")
    else:
        print(f"Gomoku - alpha-beta plays {PLAYER_NAMES[agent_side]}")
    print(f"Black = X, White = O, empty = {SYMBOLS[EMPTY]}")
    print("Five or more in a horizontal, vertical, or diagonal line wins.")
    print("Enter q to quit at any prompt.\n")
    game: Gomoku | None = None

    try:
        while True:
            try:
                size = parse_size(_read_input("Board size (9, 11, or 15): "))
                break
            except ValueError as error:
                print(f"Invalid board size: {error}")

        game = Gomoku(size)
        agent = AlphaBetaAgent() if agent_side is not None else None
        print(f"\nEnter a column and row separated by a space, each from 1 to {size}.")
        print("Count columns from the left and rows from the top. Black moves first.\n")
        print(game.render())

        while not game.game_over:
            name = PLAYER_NAMES[game.current_player]
            symbol = SYMBOLS[game.current_player]
            player = f"{name} ({symbol})"
            note = ""
            if agent is not None and game.current_player == agent_side:
                print(f"\n{name} ({symbol}, AI) is thinking. Press Ctrl+C to stop.", flush=True)
                x, y = agent.choose_move(game)
                game.play(x, y)
                player = f"{name} ({symbol}, AI)"
                note = f" [{agent.last_reason}]"
            else:
                try:
                    x, y = parse_move(_read_input(f"\n{player}, enter a spot (column row): "))
                    game.play(x, y)
                except ValueError as error:
                    print(f"Invalid move. {error}")
                    continue

            print(f"\n{player} played ({x}, {y}){note}.")
            print(game.render())

        if game.winner is not None:
            print(f"\n{PLAYER_NAMES[game.winner]} ({SYMBOLS[game.winner]}) wins!")
        else:
            print("\nDraw! The board is full and neither player has five in a row.")
        print(f"Game over after {game.move_count} moves.")
    except (EOFError, KeyboardInterrupt):
        print("\nGame stopped.")
        if game is not None:
            print(game.render())

    return 0


if __name__ == "__main__":
    colors = {"black": BLACK, "white": WHITE}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=colors, help="color played by the alpha-beta agent")
    args = parser.parse_args()
    raise SystemExit(main(agent_side=colors.get(args.agent)))
