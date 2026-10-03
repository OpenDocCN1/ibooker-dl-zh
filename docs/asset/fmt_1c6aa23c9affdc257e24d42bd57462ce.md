
```python
def minibatch_train(use_extrinsic=True):
    state1_batch, action_batch, reward_batch, state2_batch = replay.get_batch() 
    action_batch = action_batch.view(action_batch.shape[0],1)               # *1*
    reward_batch = reward_batch.view(reward_batch.shape[0],1)

    forward_pred_err, inverse_pred_err = ICM(state1_batch, action_batch, state2_batch)  # *2*
    i_reward = (1. / params['eta']) * forward_pred_err                      # *3*
    reward = i_reward.detach()                                              # *4*
    if use_explicit:                                                        # *5*
        reward += reward_batch 
    qvals = Qmodel(state2_batch)                                            # *6*
    reward += params['gamma'] * torch.max(qvals)
    reward_pred = Qmodel(state1_batch)
    reward_target = reward_pred.clone()
    indices = torch.stack( (torch.arange(action_batch.shape[0]), \          # *7*
    action_batch.squeeze()), dim=0)
    indices = indices.tolist()
    reward_target[indices] = reward.squeeze()
    q_loss = 1e5 * qloss(F.normalize(reward_pred), \
    F.normalize(reward_target.detach()))
    return forward_pred_err, inverse_pred_err, q_loss
```

- **1** 将这些张量重塑以添加一个维度，使其与模型兼容。
- **2** 运行 ICM。
- **3** 使用 `eta` 参数缩放前向预测误差。
- **4** 开始累计奖励；确保分离 `i_reward` 张量。
- **5** `use_explicit` 布尔变量让我们决定是否在内在奖励之外使用显式奖励。
- **6** 计算下一个状态的动作值。
- **7** 由于 `action_batch` 是一个动作索引的整数张量，我们将其转换为 one-hot 编码向量的张量。

现在，让我们处理主要的训练循环，如列表 8.12 所示。我们使用之前定义的 `prepare_initial_state(...)` 函数初始化第一个状态，该函数只是将第一帧重复三次沿通道维度。我们还设置了一个 `deque` 实例，我们将逐帧将其附加到该实例。`deque` 设置为 `maxlen` 为 `3`，因此只存储最近的三个帧。我们在将其传递到 Q 网络之前，将 `deque` 首先转换为列表，然后转换为维度为 `1 × 3 × 42 × 42` 的 PyTorch 张量。

##### 列表 8.12. 训练循环

```python
epochs = 3500
env.reset()
state1 = prepare_initial_state(env.render('rgb_array'))
eps=0.15
losses = []
episode_length = 0
switch_to_eps_greedy = 1000
state_deque = deque(maxlen=params['frames_per_state'])
e_reward = 0.
last_x_pos = env.env.env._x_position                             # *1*
ep_lengths = []
use_explicit = False
for i in range(epochs):
    opt.zero_grad()
    episode_length += 1
    q_val_pred = Qmodel(state1)                                  # *2*
    if i > switch_to_eps_greedy:                                 # *3*
        action = int(policy(q_val_pred,eps))
    else:
        action = int(policy(q_val_pred))
    for j in range(params['action_repeats']):                    # *4*
        state2, e_reward_, done, info = env.step(action)
        last_x_pos = info['x_pos']
        if done:
            state1 = reset_env()
            break
        e_reward += e_reward_
        state_deque.append(prepare_state(state2))
    state2 = torch.stack(list(state_deque),dim=1)                # *5*
    replay.add_memory(state1, action, e_reward, state2)          # *6*
    e_reward = 0
    if episode_length > params['max_episode_len']:               # *7*
        if (info['x_pos'] - last_x_pos) < params['min_progress']:
            done = True
        else:
            last_x_pos = info['x_pos']
    if done:
        ep_lengths.append(info['x_pos'])
        state1 = reset_env()
        last_x_pos = env.env.env._x_position
        episode_length = 0
    else:
        state1 = state2
    if len(replay.memory) < params['batch_size']:
        continue
    forward_pred_err, inverse_pred_err, q_loss = minibatch_train(use_extrinsic=False)  # *8*
    loss = loss_fn(q_loss, forward_pred_err, inverse_pred_err)   # *9*
    loss_list = (q_loss.mean(), forward_pred_err.flatten().mean(),\
                      inverse_pred_err.flatten().mean())
    losses.append(loss_list)
    loss.backward()
    opt.step()
```

- **1** 我们需要跟踪最后 x 位置，以便在没有前进进度的情况下重置。
- **2** 运行 DQN 以获取动作值预测。
- **3** 在前 1,000 个 epoch 之后，切换到 epsilon-greedy 策略。
- **4** 重复执行策略指定的任何动作 6 次，以加快学习速度。
- **5** 将 `deque` 对象转换为张量。
- **6** 将单个经验添加到重放缓冲区。
- **7** 如果马里奥没有取得足够的向前进度，重新开始游戏并再次尝试。
- **8** 从重放缓冲区获取一个数据小批次的错误。
- **9** 计算总体损失。

虽然这个训练循环有点长，但它相当简单。我们只是准备一个状态，输入到 DQN 中，获取动作值（Q 值），输入到策略中，获取要采取的动作，然后调用 `env.step(action)` 方法执行该动作。然后我们获取下一个状态和一些其他元数据。我们将整个经验作为一个元组 `(S[t], a[t], r[t], S[t]+1)` 添加到经验重放内存中。大部分动作发生在我们之前已经覆盖的小批量训练函数中。

这就是构建用于在超级马里奥兄弟上训练的端到端 DQN 和 ICM 所需的主要代码。让我们通过训练 5,000 个 epoch 来测试它，这大约需要 30 分钟的时间，在运行没有 GPU 的 MacBook Air 上。我们在小批量函数中使用 `use_extrinsic=False` 进行训练，因此它只从内在奖励中学习。你可以使用以下代码绘制每个 ICM 组件和 DQN 的单独损失。我们将对损失数据进行对数变换，以保持它们在相似的尺度上。

```python
>>> losses_ = np.array(losses)
>>> plt.figure(figsize=(8,6))
>>> plt.plot(np.log(losses_[:,0]),label='Q loss')
>>> plt.plot(np.log(losses_[:,1]),label='Forward loss')
>>> plt.plot(np.log(losses_[:,2]),label='Inverse loss')
>>> plt.legend()
>>> plt.show()
```

如图 8.17 所示，DQN 损失最初下降，然后缓慢增加并趋于平稳。前向损失似乎缓慢下降，但相当嘈杂。逆模型看起来有点平坦，但如果你放大查看，它似乎确实随着时间的推移而非常缓慢地下降。如果你设置 `use_extrinsic=True` 并使用外部奖励，损失图看起来会更好。但不要因为损失图而感到失望。如果我们测试训练好的 DQN，你会看到它的表现比损失图所暗示的要好得多。这是因为 ICM 和 DQN 像对抗性动态系统一样表现，因为前向模型试图降低其预测误差，而 DQN 试图通过将智能体引导到环境中的不可预测状态来最大化预测误差（图 8.18）。

##### 图 8.17. 这是 ICM 和 DQN 的各个组件的损失。损失并没有像我们习惯的单个监督神经网络那样平滑地下降，因为 DQN 和 ICM 是进行对抗性训练的。

![图 8.17](img/08fig17_alt.jpg)

##### 图 8.18. DQN 智能体和前向模型试图优化对抗性目标，因此形成了一个对抗性对。

![图 8.18](img/08fig18_alt.jpg)

如果你查看*生成对抗网络*（GAN）的损失图，生成器和判别器的损失看起来与我们的 DQN 和前向模型损失 `use_extrinsic=False` 时有些相似。损失并没有像你训练单个机器学习模型时那样平滑地下降。

对整体训练进展的更好评估是跟踪随时间变化的场景长度。如果智能体正在学习如何更有效地通过环境，场景长度应该会增加。在我们的训练循环中，每当场景结束时（即，当 `done` 变量变为 `True`，因为智能体死亡或没有足够的向前进展时），我们将当前的 `info['x_pos']` 保存到 `ep_lengths` 列表中。我们预计最大场景长度会随着训练时间的推移而越来越长。

```python
>>> plt.figure()
>>> plt.plot(np.array(ep_lengths), label='Episode length')
```

在图 8.19 中，我们可以看到，一开始最大的峰值达到 150（即游戏中的 x 位置），但随着训练时间的推移，智能体能够达到的最远距离（由峰值的高度表示）稳步增加，尽管有一些随机性。

##### 图 8.19. 训练时间在 x 轴上，场景长度在 y 轴上。我们看到随着训练时间的推移，峰值越来越大，这正是我们所期望的。

![图 8.19](img/08fig19_alt.jpg)

集剧长度图看起来很有希望，但让我们渲染一个我们训练的智能体玩超级马里奥兄弟的视频。如果您在自己的计算机上运行此代码，OpenAI Gym 提供了一个渲染函数，该函数将打开一个新窗口，显示实时游戏。不幸的是，如果您使用远程机器或云虚拟机，这将不起作用。在这种情况下，最简单的替代方案是运行一个游戏循环，将每个观察帧保存到列表中，一旦循环终止，就将其转换为 numpy 数组。然后您可以将这个视频帧的 numpy 数组保存为视频，并在 Jupyter Notebook 中播放。

```python
>>> import imageio; 
>>> from IPython.display import Video; 
>>> imageio.mimwrite('gameplay.mp4', renders, fps=30); 
>>> Video('gameplay.mp4')
```

在列表 8.13 中，我们使用内置的 OpenAI Gym 渲染方法来实时查看游戏。

##### 列表 8.13. 测试训练好的智能体

```python
eps=0.1
done = True
state_deque = deque(maxlen=params['frames_per_state'])
for step in range(5000):
    if done:
        env.reset()
        state1 = prepare_initial_state(env.render('rgb_array'))
    q_val_pred = Qmodel(state1)
    action = int(policy(q_val_pred,eps))
    state2, reward, done, info = env.step(action)
    state2 = prepare_multi_state(state1,state2)
    state1=state2
    env.render()
env.close()
```

如果您遵循了训练循环，这里就没有太多需要解释的；我们只是提取了运行网络前向传播并采取行动的部分。请注意，我们仍然使用 epsilon-greedy 策略，其中 epsilon 设置为 0.1。即使在推理过程中，智能体也需要一点随机性以避免陷入停滞。一个值得注意的差异是，在测试（或推理）模式下，我们只执行一次动作，而不是像训练中那样执行六次。假设您得到的结果与我们相同，您的训练智能体应该会有相当一致的前进进度，并且能够跳过障碍物（图 8.20）。恭喜！

##### 图 8.20. 仅从内在奖励训练的马里奥智能体成功跳过了一个深渊。这表明它已经学会了基本的技能，而无需任何明确的奖励来做这件事。使用随机策略，智能体甚至无法向前移动，更不用说学会跳过障碍物了。

![](img/08fig20.jpg)

如果您没有得到相同的结果，尝试更改超参数，特别是学习率、小批量大小、最大剧集长度和最小前进进度。使用内在奖励进行 5,000 个时期的训练是有效的，但根据我们的经验，它对这些超参数很敏感。当然，5,000 个时期并不长，所以进行更长时间的训练将导致更有趣的行为。

---

**在其他环境中这将如何工作？**

我们在一个环境中使用基于 ICM 的奖励训练了我们的 DQN 智能体，这个环境是超级马里奥兄弟，但 Yuri Burda 等人于 2018 年发表的论文“Curiosity-Driven Learning 的大规模研究”展示了仅使用内在奖励本身可以多么有效。他们使用基于好奇心的奖励在多个游戏中进行了一系列实验，发现一个好奇的智能体可以在超级马里奥兄弟中通过 11 个关卡，并且可以学会玩乒乓球等游戏。他们基本上使用了我们刚刚构建的相同 ICM，但他们使用了一个更复杂的演员-评论家模型，称为**近端策略优化**（PPO），而不是 DQN。

你可以尝试的一个实验是将编码器网络替换为**随机投影**。随机投影仅仅意味着将输入数据乘以一个随机初始化的矩阵（例如，一个固定且未训练的随机初始化神经网络）。Burda 等人 2018 年的论文证明了随机投影的效果几乎与训练过的编码器一样好。

---

### 8.7. 替代内在奖励机制

在本章中，我们描述了 RL 代理在稀疏奖励环境中面临的严重问题。我们认为解决方案是赋予代理一种好奇心，并实施了 Pathak 等人 2017 年论文中的方法，这是近年来强化学习研究中最广泛引用的论文之一。我们选择展示这种方法，不仅因为它很受欢迎，而且因为它建立在前面章节学到的内容之上，而没有引入太多新概念。基于好奇心的学习（有多个名称）是一个非常活跃的研究领域，有许多替代方法，其中一些我们认为比 ICM 更好。

许多其他令人兴奋的方法使用贝叶斯推理和信息论来提出新的机制来驱动好奇心。我们在本章中使用的预测误差（PE）方法只是更广泛的 PE 伞下的一种实现。正如你现在所知，基本思想是代理想要减少其 PE（或者说，其对环境的不确定性），但它必须通过积极寻求新异来做到这一点，以免被意外之事所惊吓。
