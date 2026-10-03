
```
def is_move_self_capture(self, player, move):
    if not move.is_play:
        return False
    next_board = copy.deepcopy(self.board)
    next_board.place_stone(player, move.point)
    new_string = next_board.get_go_string(move.point)
    return new_string.num_liberties == 0
```

#### 3.2.2. 提子

在检查了自杀之后，你现在可以继续实现提子规则。第二章简要介绍了提子规则及其在围棋游戏中的重要性。简而言之，如果一步棋会使棋盘回到精确的先前位置，则适用提子规则。这并不意味着玩家不能立即反击，以下图示序列显示了这一点。在图 3.5 中，白方刚刚在底部落下一颗孤子。现在黑方的两颗棋子只剩下一个眼位——但白方的这颗棋子也是如此。

##### 图 3.5。在这种情况下，白方想要捕获两颗黑棋，但白方的棋子只剩下一个眼位。

![图片](img/03fig05.jpg)

黑方现在可以尝试通过捕获这颗白棋来挽救其两颗棋子，如图 3.6 所示。

##### 图 3.6。继续，黑方试图通过捕获孤立的这颗白棋来挽救其两颗棋子。

![图片](img/03fig06.jpg)

但白方可以立即在图 3.5 中落子的相同点上落子，如图 3.7 所示。

你可以看到白方可以立即夺回三个黑子，劫争规则不适用，因为图 3.5（#ch03fig05）和图 3.7（#ch03fig07）中的整体棋盘位置不同。这种玩法被称为*立即夺回*。在简单情况下，劫争规则归结为不能立即夺回一个棋子。但立即夺回是常见的，这种位置的存在表明在实现劫争时你必须小心。

##### 图 3.7\. 在这种情况下，白方可以简单地立即收回（夺回三个黑子）而不违反劫争规则。

![图片](img/03fig07.jpg)

你可以通过许多方式指定劫争规则，但这些方式在罕见情况下实际上等效。你将在代码中实施的规则是，玩家不得下出一个会重新创建先前游戏状态的棋子，其中游戏状态包括棋盘上的棋子和下一个回合的玩家。这种特定的表述被称为*情境超劫*规则。

因为每个`GameState`实例都保留了对上一个状态的指针，你可以通过向上遍历树并检查新状态与整个历史记录来实施劫争规则。你通过向你的`GameState`实现添加以下方法来完成。

##### 列表 3.13\. 当前游戏状态是否违反了劫争规则？

```
@property
def situation(self):
    return (self.next_player, self.board)

def does_move_violate_ko(self, player, move):
    if not move.is_play:
        return False
    next_board = copy.deepcopy(self.board)
    next_board.place_stone(player, move.point)
    next_situation = (player.other, next_board)
    past_state = self.previous_state
    while past_state is not None:
        if past_state.situation == next_situation:
            return True
        past_state = past_state.previous_state
    return False
```

这种实现简单且正确，但相对较慢。对于每个移动，你都会创建一个棋盘状态的深拷贝，并必须将此状态与所有先前状态进行比较，这会随着时间的推移而累积。在第 3.5 节（#ch03lev1sec5）中，你将遇到一种有趣的技术来加快这一步骤。

为了完成你的`GameState`定义，你现在可以使用关于劫争和自我捕获的知识来决定一个移动是否有效。

##### 列表 3.14\. 在给定的游戏状态下，这个移动是否有效？

```
def is_valid_move(self, move):
    if self.is_over():
        return False
    if move.is_pass or move.is_resign:
        return True
    return (
        self.board.get(move.point) is None and
        not self.is_move_self_capture(self.next_player, move) and
        not self.does_move_violate_ko(self.next_player, move))
```

### 3.3\. 结束游戏

在计算机围棋中，一个关键概念是*自我对弈*。在自我对弈中，你通常从一个弱的围棋代理开始，让它与自己对弈，并使用游戏结果来构建一个更强的机器人。在第四章（kindle_split_016.xhtml#ch04）中，你将使用自我对弈来评估棋盘位置。在第九章（kindle_split_021.xhtml#ch09）到第十二章（kindle_split_024.xhtml#ch12）中，你将使用自我对弈来评估单个移动及其选择的算法。

为了利用这项技术，你需要确保你的自我对弈游戏结束。人类游戏在双方都无法通过下一步移动获得优势时结束。即使对于人类来说，这也是一个棘手的概念。初学者经常通过在对手领土上玩无望的移动或观察对手切入他们认为稳固的领土来结束游戏。对于计算机来说，这更加困难。如果我们的机器人只要还有合法的移动就继续玩，它最终会填满自己的领地并失去所有的棋子。

你可以想出一些启发式方法来帮助机器人以合理的方式结束游戏。例如：

+   不要在一个完全被同色棋子包围的区域下棋。

+   不要下只有一个眼位的棋子。

+   总是用只有一个眼位的情况下吃掉对方的棋子。

不幸的是，*所有这些规则都过于严格*。如果我们的机器人遵循这些规则，强大的对手会利用这些规则杀死应该活着的组，拯救应该死亡的组，或者简单地获得更好的位置。一般来说，我们手工编写的规则应该尽可能限制机器人的选项，以便更复杂的算法可以自由学习高级策略。

要解决这个问题，您可以参考游戏的悠久历史。在古代，胜利者仅仅是棋盘上棋子最多的玩家。玩家们会通过填满他们能填的点来结束游戏，只留下他们组的眼位为空。这可能会使游戏的结束拖得很长，因此玩家们想出了一个加快游戏速度的方法。如果黑方明显控制了棋盘上的某个区域（任何白方在那里放的棋子最终都会被吃掉），那么黑方就没有必要用棋子填满那个区域。玩家们只需同意为黑方计算那个区域。这就是领地概念的产生，几个世纪以来，规则逐渐演变，以至于领地被明确计算。

这种方法计分避免了关于什么是领地，什么不是领地的问题，但您仍然需要防止您的机器人杀死自己的棋子；参见图 3.8。

##### 图 3.8\. 白方在角落 A 和 B 有两个眼位，不应在这两个点放置棋子。否则，黑方可以吃掉整个组。您的天真机器人将不允许填充自己的眼位。

![图片](img/03fig08.jpg)

您将硬编码一个规则，以防止机器人按照最严格的定义填充自己的眼位。就我们的目的而言，一个*眼位*是一个空点，其中所有相邻的点以及至少三个四对角相邻的点都填满了友军棋子。

| |

| --- |

##### 注意

经验丰富的围棋玩家可能会注意到，前面关于*眼位*的定义在某些情况下会错过一个有效的眼位。我们将接受这些错误以保持实现简单。

| |

| --- |

您必须为棋盘边缘的眼睛创建一个特殊情况；在这种情况下，所有对角相邻的点都必须包含友军棋子。

您创建一个名为*agent*的新 dlgo 子模块（通过创建一个名为*agent*的新文件夹，并在该文件夹内创建一个空的`__init__.py`文件）并将以下`is_point_an_eye`函数放入一个名为`helpers.py`的文件中。

##### 列表 3.15\. 给定的点是否是眼位？

```
from dlgo.gotypes import Point

def is_point_an_eye(board, point, color):
    if board.get(point) is not None:                       *1*
        return False
    for neighbor in point.neighbors():                     *2*
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False

    friendly_corners = 0                                   *3*
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4   *4*
    return friendly_corners >= 3                           *5*
```

+   ***1* 眼位是一个空点。**

+   ***2* 所有相邻的点都必须包含友军棋子。**

+   ***3* 如果点位于棋盘中间，我们必须控制四个角落中的三个；在边缘，你必须控制所有角落。**

+   ***4* 点位于边缘或角落。**

+   ***5* 点位于中间。**

在本章中，你还没有明确关注确定游戏结果，但计算游戏结束时的分数绝对是一个重要的话题。不同的锦标赛和围棋联合会执行略有不同的规则集。在整个书中，你的机器人将遵循 AGA 的*面积计数*规则，也称为*中国计数*。尽管*日本规则*在休闲游戏中更为流行，但 AGA 规则对计算机来说更容易一些，规则差异很少影响游戏结果。

### 3.4\. 创建你的第一个机器人：想象中最弱的围棋 AI

完成了围棋盘的实现和游戏状态的编码后，你可以构建你的第一个围棋机器人。这个机器人将是一个弱手，但它将为所有后续改进的机器人打下基础。首先，你定义所有机器人将遵循的接口。你将代理的定义放入`agent`模块中的`base.py`文件中。

##### 列表 3.16\. 围棋代理的中心接口

```
class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
```

就这么简单，只有一个方法。机器人所做的只是根据当前游戏状态选择一个移动。当然，在内部这可能需要其他复杂的任务，例如评估当前位置，但为了玩游戏，我们的机器人永远只需要这些。

我们的第一种实现将尽可能简单：它将随机选择任何有效的移动，但不会填充自己的一个眼位。如果没有这样的移动，它将放弃。你将这个随机机器人放入`agents`目录下的`naive.py`文件中。回想一下第二章，围棋的学生段位通常从 30 段位到 1 段位不等。按照这个标准，你的随机机器人处于 30 段位水平，这是一个绝对的初学者。

##### 列表 3.17\. 一个随机围棋机器人，大约 30 段位水平

```
import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point

class RandomBot(Agent):
    def select_move(self, game_state):
        """Choose a random valid move that preserves our own eyes."""
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(Move.play(candidate)) and \
                        not is_point_an_eye(game_state.board,
                                            candidate,
                                            game_state.next_player):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
```

到目前为止，你的模块结构应该如下所示（确保在文件夹中放置一个空的`__init__.py`文件以初始化子模块）：

```
dlgo
  ...
  agent
    __init__.py
    helpers.py
    base.py
    naive.py
```

最后，你可以设置一个驱动程序，让两个随机机器人实例之间进行完整游戏。首先，你定义方便的辅助函数，例如在控制台上打印整个棋盘或单个移动。

围棋盘坐标可以用多种方式指定，但在欧洲，最常见的是用字母表中的字母标记列，从 A 开始，行用递增的数字标记，从 1 开始。在这些坐标中，在一个标准的 19×19 棋盘上，左下角是 A1，右上角是 T19。请注意，按照惯例省略了字母 I，以避免与数字 1 混淆。

你定义了一个字符串变量`COLS = 'ABCDEFGHJKLMNOPQRST'`，其中的字符代表围棋棋盘的列。为了在命令行上显示棋盘，你用一个点(`.`)表示空场，用`x`表示黑子，用`o`表示白子。以下代码放入我们称为`utils.py`的新文件中，在`dlgo`包中。你创建了一个`print_move`函数，用于将下一个移动打印到命令行，以及一个`print_board`函数，用于打印带有所有石头的当前棋盘。你将此代码放入`dlgo`模块外的名为`bot_v_bot.py`的文件中。

##### 列表 3.18\. 对战机器人游戏的实用函数
