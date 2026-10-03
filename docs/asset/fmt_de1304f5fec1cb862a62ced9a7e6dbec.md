
### 列表 3.26. 使用 Zobrist 哈希初始化游戏状态

```
class GameState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.zobrist_hash())})
        self.last_move = move
```

如果棋盘为空，`self.previous_states` 是一个空的不可变 `frozenset`。否则，你通过一对来增强状态：下一个玩家的颜色和上一个游戏状态的 Zobrist 哈希。

在设置好所有这些之后，你最终可以大幅改进你的 `does_move_violate_ko` 实现。

### 列表 3.27. 使用 Zobrist 哈希快速检查 ko 状态

```
    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states
```

通过单行 `next_situation in self.previous_states` 检查之前的棋盘状态比之前显式遍历棋盘状态的效率高了一个数量级。

这个有趣的哈希技巧将使后续章节中的自我对弈更快，从而在游戏玩法上带来更快的改进。

## 进一步加速你的围棋盘实现

我们对原始的 `goboard_slow.py` 实现进行了深入分析，并展示了如何通过 Zobrist 哈希来加速它，从而得到 `goboard.py`。在 GitHub 仓库中，你还会看到另一个名为 `goboard_fast.py` 的围棋盘实现，其中你可以进一步加快游戏速度。这些速度提升在后续章节中非常有价值，但这是以可读性为代价的。

如果你感兴趣，想了解如何使你的围棋盘更快，请查看 `goboard_fast.py` 以及那里的注释。大多数优化都是避免构造和复制 Python 对象的技巧。

## 3.6. 与你的机器人对抗

在创建了一个与自己对抗的弱机器人之后，你可能想知道你是否可以测试你的知识从第二章开始，自己与它对抗。这确实可能，并且与机器人对抗机器人的设置相比，不需要太多改变。

你还需要一个额外的实用函数，将其放入 `utils.py` 中，以帮助你从人类输入中读取坐标。

### 列表 3.28. 将人类输入转换为围棋盘坐标

```
def point_from_coords(coords):
    col = COLS.index(coords[0]) + 1
    row = int(coords[1:])
    return gotypes.Point(row=row, col=col)
```

这个函数将像 `C3` 或 `E7` 这样的输入转换为围棋盘坐标。考虑到这一点，你现在可以在 `human_v_bot.py` 中设置程序以进行 9 × 9 游戏如下。

### 列表 3.29. 设置脚本以便你可以玩自己的机器人

```
from dlgo import agent
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input

def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = agent.RandomBot()

    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)

if __name__ == '__main__':
    main()
```

你，作为人类玩家，将扮演黑方。你的随机机器人扮演白方。用以下方式启动脚本：

```
python human_v_bot.py
```

你将被提示输入一个移动，并通过 `enter` 确认。例如，如果你选择将 `G3` 作为你的第一个移动，机器人的响应可能如下所示：

```
Player.white D8
9 .........
8 ...o.....
7 .........
6 .........
5 .........
4 .........
3 ......x..
2 .........
1 .........
  ABCDEFGHJ
```

如果你愿意，你可以继续与这个机器人进行完整游戏。但由于机器人玩的是随机移动，目前这样做并不有趣。

注意，在遵循围棋规则方面，这个机器人已经完整了。它知道它将需要的所有关于围棋的知识。这个事实很重要，因为它让你可以完全专注于从现在开始提高游戏算法。这个机器人代表了你的起点。在接下来的章节中，你将介绍更多有趣的技巧来创建更强大的机器人。

## 3.7. 摘要

+   围棋游戏中两位玩家的最佳编码方式是使用枚举。

+   围棋棋盘上的一个点由其直接邻域来表征。

+   围棋中的一步棋可以是落子、认输或投降。

+   石子串是由相同颜色的石子组成的连接组。石子串对于放置石子后高效检查被捕获的石子非常重要。

+   围棋的 `Board` 包含了放置和捕获石子的所有逻辑。

+   相比之下，`GameState` 跟踪的是轮到谁走，当前棋盘上的石子，以及之前状态的历史。

+   Ko 可以通过使用情境超级 ko 规则来实现。

+   Zobrist 哈希是一种重要的技术，可以高效地编码游戏历史并加快 ko 的检查。

+   一个围棋玩棋代理可以通过一个方法来定义：`select_move`。

+   您的随机机器人可以与自己、其他机器人以及人类对手进行对弈。
