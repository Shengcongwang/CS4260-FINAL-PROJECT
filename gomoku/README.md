# Gomoku

A terminal Gomoku game with two-player mode, an alpha-beta opponent, and a
reusable Python game engine. Requires Python 3.9+ and NumPy.

## Run

From PowerShell:

```powershell
cd 'D:\Vandy\CS4260\final project\gomoku'
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe play.py
```

If your Python environment already has NumPy, run `python play.py` instead.
Once the environment is set up, only the final command is needed to play again.
To play against the alpha-beta agent, add `--agent white` or `--agent black`.

```powershell
.\.venv\Scripts\python.exe play.py --agent white
```

1. Choose `9`, `11`, or `15`. Dimensions such as `9x9` and `15*15` also work.
2. Empty spots appear as `+`. Black (`X`) moves first, followed by White (`O`).
3. Enter a move such as `3 5` for column 3, row 5. Count from **1**, starting
   at the leftmost column and the top row.
4. After each legal move, the board is displayed. A human player gets a prompt,
   and the agent automatically plays its turn and displays why it chose the move.
   Malformed, out-of-bounds, or occupied coordinates leave both board and turn
   unchanged, and the same player tries again.
5. Five **or more** consecutive stones horizontally, vertically, or along either
   diagonal wins. The final board and winning player are printed. A full board
   without a winner is a draw. Enter `q`, `quit`, or `exit` to stop early.

Players may place a stone on any empty spot. The two-square neighborhood
restriction applies to the agent's search candidates.

## Files

- `game.py`: NumPy board, move validation, turn tracking, win/draw checks, and rendering.
- `play.py`: interactive board-size selection and human or agent turns.
- `threats.py`: detection and blocking of the opponent's contiguous threats.
- `alphabeta.py`: alpha-beta minimax with a ten-move horizon.
- `rollouts.py`: sampled continuations for estimating loss rates.
- `tests/`: game-rule and terminal-input regression tests.

## How the alpha-beta agent plays

The agent checks the opponent's horizontal, vertical, and diagonal lines in
this priority order. An open end is an empty cell on the board.

1. **Alive four.** First check nearby empty cells for any move that would give
   the opponent five or more in a row. Block that cell, including when it fills
   a gap in the line.
2. **Open four.** Block one open end of four consecutive stones with at least
   one open end.
3. **Alive three.** Block one end of three consecutive stones with both ends open.
4. **Double three.** Occupy a spot where the opponent could create two alive
   threes, an alive three and an open four, or two open fours. Each qualifying
   line must pass through the new stone.

An **open three** is three consecutive stones with exactly one open end. The
alive-three and double-three rules require both ends open for a run of three.

If several cells would let the opponent win, the agent blocks the one closest
to the center. At lower priorities it favors a spot that addresses more
threats, then one closer to the center. An open four with two open ends still
has a winning continuation at the other end after one block.

On an empty board the agent plays the center. If there is no block to make, it
takes an immediate win when available. Otherwise, alpha-beta search looks ahead
at most ten individual moves, including the agent's candidate move. It checks
every candidate within two squares of a stone, including diagonals. To keep the
search manageable, it examines up to three promising replies at each deeper
turn. With ten or fewer empty cells it examines every reply that alpha-beta
cannot prune, and the game will end within the horizon.

For longer positions, the agent samples 16 continuations per candidate. Both
players take immediate wins or blocks when available and otherwise choose a
nearby move at random. An opponent win within ten moves counts as a loss. Draws
and positions that are still open at ten moves count in the denominator but
not as losses. Among moves with the same minimax outcome, the agent prefers
the lower sampled loss rate. The terminal displays the estimate for a selected
move that scored as a draw or remained unresolved in the search.

## Call the agent from Python

The agent returns an `(x, y)` tuple using one-based coordinates. Use
`game.play(x, y)` to validate and apply it.

```python
from alphabeta import AlphaBetaAgent
from game import Gomoku

game = Gomoku(9)
agent = AlphaBetaAgent()
game.play(*agent.choose_move(game))
print(game.render())
```

Useful engine members:

| Member | Meaning |
| --- | --- |
| `board` | NumPy `int8` array: `0` empty, `1` black, `-1` white; indexed `[y-1, x-1]` |
| `current_player` | Player to move; stays at the final mover once the game ends |
| `play(x, y)` | Applies a legal move; raises `InvalidMove` without changing state otherwise |
| `legal_moves()` | All legal `(x, y)` tuples; an empty list after game end |
| `copy()` | Independent game state for simulation |
| `game_over` | True after a win or draw |
| `winner` | `BLACK`, `WHITE`, or `None` |
| `result` | `+1` black win, `-1` white win, `0` draw, `None` ongoing |
| `move_count`, `last_move` | Number of accepted moves and the most recent `(x, y)` |
| `render()` | Board text with coordinate labels |

Use `play()` to make moves; directly editing `board` bypasses turn and result
tracking. Future search algorithms may filter `legal_moves()` themselves.

## Tests

From this folder, after installing NumPy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

