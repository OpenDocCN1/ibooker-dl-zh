
```py
def play():
 '''
 Play the game until the game is over.

 The pot starts with 0 chips, and each player starts with 0 chips.

 Randomly decide whether player 1 or player 2 goes first.

 Before each turn, print three lines of information:
 1\. The number of chips in the pot
 2\. The number of chips that each player has
 3\. Whether it is player 1's turn or player 2's turn

 Take a full turn for the current player. 
 If they won the chips, add the chips in the pot to the 
 total for that player and reset the pot to have 0 chips.

 Then, switch to the other player's turn.

 Once the game is over, print the current player 
 (that's the player who won).
 '''
    pot_chips = 0      #1
    player1 = 0        #1
    player2 = 0        #1
    current_player = random.randint(1, 2)      #2
    rolls = []
    while not game_over(player1, player2, rolls):     #3
        print('Pot chips:', pot_chips)               #4
        print('Player 1 chips:', player1)              #4
        print('Player 2 chips:', player2)            #4
        print('Player', current_player, 'turn')      #4
        pot_chips, rolls = take_full_turn(pot_chips)     #5
        if wins_chips(rolls):                           #6
            if current_player == 1:                   #7
                player1 += pot_chips                       #8
            else:
                player2 += pot_chips          #9
            pot_chips = 0                              #10
        current_player = switch_player(player1, player2,  #11
                                       rolls, current_player)
    print('Player', current_player, 'wins!')   #12
```

#1 开始筹码池和玩家，初始筹码为 0

#2 随机让玩家 1 或玩家 2 开始游戏

#3 当游戏尚未结束时...

#4 ...打印当前游戏状态。

#5 让当前玩家进行回合

#6 如果当前玩家赢得筹码...

#7 ...然后轮到玩家 1...

#8 ...将筹码池中的筹码给玩家 1；...

#9 ...否则，将筹码池中的筹码给玩家 2。

#10 将筹码池重置为 0 筹码

#11 切换到另一个玩家的回合

#12 打印出获胜者

我们现在拥有了游戏所需的所有代码。只需添加

```py
play()
```

在你现有的所有代码下方添加调用我们的`play`函数，然后你就可以玩游戏了！

## 自定义游戏

我们对 Copilot 给出的结果很满意，游戏当然可以玩。但是，诚实地讲，与玩家的互动可以更友好一些。例如，当我们运行游戏并多次按下 y 时，游戏可能会这样开始：

筹码池筹码：0

玩家 1 筹码：0

玩家 2 筹码：0

玩家 2 回合

4

Continue? (y/n) y

5

2

Continue? (y/n) y

3

1

4

筹码池筹码：6

玩家 1 筹码：0

玩家 2 筹码：0

玩家 1 回合

2

继续吗？(y/n)

没有欢迎信息。像 4、5、2 这样的数字只是在那里，没有任何上下文说明它们的意义。游戏会询问我们是否想要“继续？(y/n)”。继续什么？这并不清楚。

我们可以通过在我们的函数中增加`print`调用来使与玩家的互动更加愉快。我们只需自己将想要的内容放入每个`print`语句中即可。为什么不直接使用 Copilot 呢？这样做可能比试图说服 Copilot 打印出我们想要的确切内容要容易得多。例如，下面的列表显示了`play`的新版本，我们在其中添加了更多的`print`调用，以解释正在发生的事情，并更好地格式化游戏输出。

### 列表 10.13 Bogart 中格式更好的`play`函数

```py
def play():
 '''
 Play the game until the game is over.

 The pot starts with 0 chips, and each player starts with 0 chips.

 Randomly decide whether player 1 or player 2 goes first.

 Before each turn, print three lines of information:
 1\. The number of chips in the pot
 2\. The number of chips that each player has
 3\. Whether it is player 1's turn or player 2's turn

 Take a full turn for the current player. 
 If they won the chips, add the chips in the pot to the
 total for that player
 and reset the pot to have 0 chips.

 Then, switch to the other player's turn.

 Once the game is over, print the current player 
 (that's the player who won).
 '''
    pot_chips = 0
    player1 = 0
    player2 = 0
    current_player = random.randint(1, 2)
    rolls = []

    print('Welcome to Bogart!')       #1
    print()                            #2

    while not game_over(player1, player2, rolls):
        print('Pot chips:', pot_chips)
        print('Player 1 chips:', player1)
        print('Player 2 chips:', player2)
        print('Player', current_player, 'turn')
        pot_chips, rolls = take_full_turn(pot_chips)
        if wins_chips(rolls):
            print('Player', current_player, 'gets',  #3
                  pot_chips, 'chips!')                 #3
            if current_player == 1:
                player1 += pot_chips
            else:
                player2 += pot_chips
            pot_chips = 0
        current_player = switch_player(player1, player2, 
                                       rolls, current_player)

        print()
        print()
        print('-=' * 20)     #4
        print()

    print('Player', current_player, 'wins!')
```

#1 打印欢迎信息

#2 打印一个空行

#3 打印消息说明当前玩家获得了筹码

#4 将每个回合与下一个回合分开

我们鼓励你在其他地方也添加`print`调用，以使游戏交互更加出色！例如，在`roll_dice`中，你可以添加一个`print`调用来告诉玩家骰子即将被掷出（以下输出中的**ROLLS**）。在`take_full_turn`中，你可以添加一个`print`调用来在询问玩家是否想继续他们的回合之前打印出筹码池中的筹码数量。以下是我们在做出这些更改后与我们的游戏交互的示例：

欢迎来到 Bogart！

筹码池筹码：0

玩家 1 筹码：0

玩家 2 筹码：0

玩家 2 回合

**ROLLS**

2

筹码池目前有 1 个筹码。

继续回合？(y/n) y

**ROLLS**

6

3

筹码池目前有 3 个筹码。

继续回合？(y/n) y

**ROLLS**

1

1

6

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

筹码池筹码：6

玩家 1 筹码：0

玩家 2 筹码：0

玩家 1 回合

**ROLLS**

5

筹码池目前有 7 个筹码。

继续回合？(y/n)

也尝试对你的代码做出这样的修改。让你的游戏变得独一无二！

在本章中，我们使用 Copilot 帮助我们编写了两个电脑游戏：一个逻辑游戏（类似于 Wordle，但使用数字），和一个两人掷骰子游戏。我们能够做到这一点，得益于我们在整本书中学到的技能，包括问题分解、编写清晰的 docstrings 以及与 Copilot Chat 交互。

## 10.5 练习

1.  在以下`guess_result`函数中存在一个错误，导致正确和错误放置的数字计数不正确。找出并修复错误：

```py
def guess_result(guess, secret_code): 
   correct = 0 
   misplaced = 0 
   for i in range(len(guess)): 
      if guess[i] == secret_code[i]: 
  correct += 1 
     if guess[i] in secret_code: 
         misplaced += 1 
return [correct, misplaced]
```

2.  将本章中的`get_guess`函数修改为包含错误信息，引导玩家输入有效的猜测。确保这些信息清晰且有帮助。发挥创意！以下是一些示例：

    - “错误：猜测必须只包含数字。”
    - “错误：猜测不得包含重复的数字。”

3.  为什么我们选择将数字的位数和猜测的次数作为参数传递给 Bulls and Cows 游戏的`play`函数？这个决定如何提高游戏的灵活性？

4.  实现一个简单的掷骰子游戏，玩家掷两个骰子。如果总和是 7 或 11，玩家获胜。如果总和是 2、3 或 12，玩家失败。对于任何其他总和，玩家继续掷骰子，直到他们要么匹配他们的初始掷骰结果（获胜）要么掷出 7（失败）。仔细思考你如何想将主要问题分解成更小的子问题！

5.  实现一个 Bogart 游戏的版本，其中一个玩家是人类，另一个是电脑。为此，你需要想出一些电脑可以用来结束他们回合的规则。例如，电脑当然希望在他们的回合中积累足够的筹码以赢得游戏时停止掷骰子！电脑也可能希望在当前回合中能够赚取大量筹码时停止掷骰子。

6.  *《价格猜猜看》* 游戏节目是许多有趣游戏的绝佳来源。例如，你可以尝试实现名为 Any Number 的游戏：

[`priceisright.fandom.com/wiki/Any_Number`](https://priceisright.fandom.com/wiki/Any_Number)

在你的游戏中，玩家应该继续选择数字，直到揭示出其中一个奖项的所有数字；那就是玩家赢得的奖项。

7.  使用 Pygame 库创建一个简单的游戏。在这个游戏中，玩家控制一个角色左右移动以收集下落的物体。目标是尽可能多地收集物体而不要错过任何一个。你可以在这里对游戏中物体的外观灵活处理，尽情享受乐趣！作为一个潜在的扩展：你能想到如何跟踪玩家的最高分吗？

## 摘要

+   游戏有一个常见的程序流程，包括游戏设置和游戏玩法。

+   随机性是许多游戏的重要成分。

+   我们可以通过使用 random 模块中的函数来为我们的 Python 游戏添加随机性。

+   我们可以使用与本书中相同的流程实现游戏，其中问题分解起着关键作用。
