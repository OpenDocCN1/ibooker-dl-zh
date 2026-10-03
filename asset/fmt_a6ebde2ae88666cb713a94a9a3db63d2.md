
# 6.4.2 从 XGBoost 和 LightGBM 中提取树结构

## XGBoost 树结构提取

以下代码展示了如何从 XGBoost 模型中提取树结构：

```py
Number of trees: 10
Number of paths on tree: 8
Path 0: [SplitNode(
             feature='f5', 
             origin='=', 
             gain=19998.9316, 
             count=12223.75,
             threshold=0.5),
             SplitNode(
                 feature='f2',
                 origin='<',
                 gain=965.524414,
                 count=5871.5,
                 threshold=0.5
             ), 
            SplitNode(
                 feature='f13',
                 origin='<',
                 gain=66.1962891,
                 count=3756,
                 threshold=1.88965869
            ), 
            LeafNode(
                 index=7,
                 origin='<',
                 count=3528)
]
```

图 6.8 将获得的输出与 XGBoost 包本身提供的完整树的图形表示进行了比较。

![](img/CH06_F08_Ryan2.png)

图 6.8 XGBoost 的`plot_tree`输出

在模型中构建的 10 棵树中，代码展示了第一棵树，并在从样本到预测叶节点的 8 条不同路径中，表示了第一条路径。从视觉上看，这条路径是最左侧的。路径由一系列不同的节点组成。代码报告了使用的特征名称、从上一个节点（在 XGBoost 中，小分支总是代表左分支，而大分支等于右分支）的分割分支起源、分割阈值、相对于目标函数的增益以及根据数据集的分割导致的样本减少。所有这些信息都允许你完美地复制 XGBoost 模型中每棵树的结果。

## LightGBM 树结构提取

我们也可以从 LightGBM 中提取相同的树结构，尽管方法略有不同，因为 LightGBM 包遵循一些略微不同的约定。例如，XGBoost 总是先在左边的阈值上分割；而 LightGBM 则相反，对于每个节点，使用减号或大于等于和阈值定义一个规则，如果规则为真则在左边分割，如果为假则在右边分割。

列表 6.17 提取 LightGBM 树结构

```py
from lightgbm import LGBMClassifier, plot_tree

lgbm = LGBMClassifier(boosting_type='gbdt', 
                      n_estimators=10, 
                      max_depth=3)

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('lightgbm', lgbm)])

model_pipeline.fit(X=data, y=target_median)
model = model_pipeline["lightgbm"]

tree_info = model._Booster.dump_model()["tree_info"]           ①
tree_structure = tree_info[0]['tree_structure']                ②
plot_tree(
    booster=model._Booster,
    tree_index=0,
    dpi=600
)                                                              ③

Split = namedtuple(
    "SplitNode",
    "feature origin decision_type threshold gain count"
)
Leaf = namedtuple("LeafNode", "index origin count value")

def extract_lgbm_node_info(tree):                              ④
    return [tree['split_feature'], tree['origin'], tree['decision_type'],
            tree['threshold'], tree['split_gain'], tree['internal_count']]

def extract_lgbm_leaf_info(tree):                              ⑤
    return [
         tree['leaf_index'],
         tree['origin'],
         tree['leaf_count'], 
         tree['leaf_value']
    ]

def traverse_lgbm_tree(tree):                                  ⑥
    if not 'origin' in tree:
        tree['origin'] = ""
    if not 'left_child' in tree and not 'right_child' in tree:
        return [[Leaf(*extract_lgbm_leaf_info(tree))]]
    left_branch = tree['left_child']
    right_branch = tree['right_child']
    left_branch['origin'] = 'yes'
    right_branch['origin'] = 'no'
    left_paths = traverse_lgbm_tree(left_branch)
    right_paths = traverse_lgbm_tree(right_branch)
    node_info = [Split(*extract_lgbm_node_info(tree))]
    return [node_info + path for path in left_paths + right_paths]

paths = traverse_lgbm_tree(tree_structure)
print(paths[0])
```

① 从 LightGBM 模型增强器中提取树信息

② 从树信息中提取第一个树的结构

③ 使用`plot_tree`函数绘制集成中的第一个树

④ 从 LightGBM 树结构中的分割节点提取各种信息的函数

⑤ 从 LightGBM 树结构中的叶节点提取信息的函数

⑥ 递归遍历 LightGBM 树结构以提取路径的函数

本探索报告的结果报告了从集成中第一个决策树的结构：

```py
[SplitNode(
    feature=5, 
    origin='',
    decision_type='<=',
    threshold=1.0000000180025095e-35,
    gain=20002.19921875,
    count=48895),
 SplitNode(
    feature=2,
    origin='yes',
    decision_type='<=',
    threshold=1.0000000180025095e-35,
    gain=967.0560302734375,
    count=23486),
 SplitNode(
    feature=13,
    origin='yes',
    decision_type='<=',
    threshold=1.8896587976897459,
    gain=67.53350067138672,
    count=15024), 
 LeafNode(
    index=0,
    origin='yes',
    count=14112,
    value=-0.16892421857257725)
]
```

图 6.9 显示了`plot_tree`函数绘制的整个树，这次是从 LightGBM 包中。

![图片](img/CH06_F09_Ryan2.png)

图 6.9 LightGBM 的`plot_tree`输出

树从左到右水平绘制。我们可以检查代码返回的路径是最高路径，以叶节点 0 结束。

# 6.4.3 通过 GBDT 和编译加速

当案例数量或可用特征很多时，即使是较快的 LightGBM 也可能需要很长时间来训练此类数据上的模型。在训练时，你可以通过减小参数`subsample`的值来减少处理的案例和特征，以限制每个决策树中涉及的案例数量，以及参数`colsample_bytree`来限制在树分割时考虑的特征数量，从而克服长时间的等待。然而，减少案例或特征可能不是从你的模型中获得最佳结果的最佳选择。另一种选择是使用 GPU，因为它们广泛用于深度学习模型。GPU 可以加速训练操作，特别是与 XGBoost 一起，以及在 LightGBM 模型中，虽然程度较小但仍然显著。

## 使用 GPU 加速 XGBoost

使用 XGBoost，从建模的角度来看，使用你的 GPU 相当简单：你只需将`tree_method`参数的值指定为`"gpu_hist"`。然而，在新 2.0.0 版本中，这种方法已被弃用，用户现在可以通过`device`参数指定使用的设备。你可以将其设置为`"cpu"`以让 XGBoost 在 CPU 上执行，或者设置为`device="cuda"`以及`device="gpu"`以使其在 CUDA 支持的 GPU 上运行，目前这是唯一的选择，但将来将支持更多 GPU 类型。如果你有多个 GPU，你可以指定它们的序号来选择特定的一个；例如，`device="cuda:1"`将在你的第二个 GPU 设备上执行。

为了使 XGBoost 运行，您至少需要安装 CUDA 11.00 以及具有 5.0 计算能力的 GPU。如果您有更多的 GPU 可用，您可以通过`gpu_id`参数指定使用哪一个，该参数代表 CUDA 运行时报告的 GPU 设备序号（如果您只有一个 GPU，通常设置为 0）。这样，XGBoost 将决策树的生长移动到 GPU 内存和处理器中，从而获得相关的操作速度，特别是特征直方图，如 Mitchell 和 Frank 在论文“Using GPU Computing to Accelerate the XGBoost Algorithm”中所述（[`peerj.com/articles/cs-127/`](https://peerj.com/articles/cs-127/)）。

一旦 GPU 训练了一个模型，它就可以在具有 GPU 的机器上进行预测。您只需设置`predictor`参数为`gpu_predictor`或如果您想使用 CPU，则设置为`cpu_predictor`。当您需要计算模型可解释性的 SHAP 值和 SHAP 交互值时，选择 GPU 作为预测器参数也可以加快速度：

```py
model.set_param({"predictor": "gpu_predictor"})
shap_values = model.predict(X, pred_contribs=True)
shap_interaction_values = model.predict(X, pred_interactions=True)
```

## 使用 GPU 加速 LightGBM

虽然使用 GPU 与 XGBoost 一起使用很简单，但与 LightGBM 一起使用就变得有点复杂。LightGBM 没有 GPU 运行的选项，而是需要为其编译一个特殊版本。根据您的操作系统（Windows、Linux/Ubuntu、MacOS），编译可能更具挑战性。对于 POSIX 系统，请参阅[`mng.bz/nRg5`](https://mng.bz/nRg5)的说明，对于 Windows 系统，请参阅[`mng.bz/vK8p`](https://mng.bz/vK8p)的说明。然而，如果您已按照[`mng.bz/4aJg`](https://mng.bz/4aJg)中的说明准备好所有先决条件，您可以直接在 shell 或命令提示符中使用 pip install 指令进行安装：

```py
pip install lightgbm --install-option=--gpu
```

一切安装完成后，您需要将参数`device`设置为`gpu`。不过，不要期待惊人的性能提升。正如 LightGBM 作者所述（见[`mng.bz/vK8p`](https://mng.bz/vK8p)），在大型和密集数据集上可以获得最佳结果，因为不高效的数据周转会导致在处理小型数据集时产生延迟。此外，为直方图算法设置更少的 bins 数量将使 GPU 与 LightGBM 更有效地工作。建议将`max_bin=15`和单精度，`gpu_use_dp=false`设置为最佳性能。

## 使用 Treelite 和 lleaves 加速预测

GPU 对于加速训练非常有用，但在预测时还有更多选项。正如我们在上一节中看到的，由于树结构如此容易获得，一些特定项目已经可以使用这种信息来重建预测树，使用性能更好的编程语言，如 C、JAVA 或 LLVM，这些语言可以将你的模型转换为纯汇编代码。这样的树编译项目旨在实现快速预测和更容易部署。例如，Treelite ([`github.com/dmlc/treelite`](https://github.com/dmlc/treelite)) 可以读取由 XGBoost、LightGBM 甚至 Scikit-learn 生成的模型，还有 lleaves ([`github.com/siboehm/lleaves`](https://github.com/siboehm/lleaves))，这是一个仅针对 LightGBM 的项目。

从 Treelite 开始，这个项目致力于成为决策树森林的通用模型交换和序列化格式。它将你的 GBDT 编译成 C 或 Java，依赖性尽可能少，因此你可以轻松地将它部署到任何系统。为了进行测试，你必须在命令行中安装几个包：`pip install tl2cgen treelite treelite_runtime`。

列表 6.18 Treelite 加速 XGBoost 预测

```py
import treelite
import treelite_runtime
import tl2cgen

xgb = XGBClassifier(booster='gbtree',
                      objective='reg:logistic',
                      n_estimators=10,
                      max_depth=3)

model_pipeline = Pipeline(
     [('processing', column_transform),
     ('XGBoost', xgb)])

model_pipeline.fit(X=data, y=target_median)
model = model_pipeline["XGBoost"]

model.save_model("./xgb_model.json")                          ①
treelite_model = treelite.Model.load("./xgb_model.json", 
model_format="XGBoost_json")                                  ②
tl2cgen.generate_c_code(treelite_model, dirpath="./", 
params={"parallel_comp": 4})
tl2cgen.export_lib(treelite_model, toolchain="gcc", 
libpath="./xgb_model.so",                                     ③
                   params={"parallel_comp": 4})

predictor = tl2cgen.Predictor("./xgb_model.so")
X = model_pipeline["processing"].transform(data)              ④
dmat = tl2cgen.DMatrix(X)                                     ⑤
predictor.predict(dmat) 
```

① 将 XGBoost 模型保存到 JSON 文件

② 从 JSON 文件加载 Treelite 格式的 XGBoost 模型

③ 从 Treelite 模型生成 C 代码并将其导出为共享库

④ 使用管道中定义的预处理步骤转换输入数据

⑤ 从转换后的数据创建与导出 Treelite 模型兼容的 Treelite DMatrix

结果是一个编译后的模型，在 Python 脚本中可以以更快的速度返回预测。预测器必须在转换之前进行转换，因为管道不是编译的一部分。只有模型是。此外，你还需要在将数据发送到编译模型之前将其转换为 DMatrix 格式，这是 XGBoost 的本地数据格式。

由 Simon Boehm 开发，lleaves 通过基于可以从 LightGBM 模型输出的文本树结构使用 LLVM 编译到汇编，承诺实现 x10 的速度提升。通过在命令行上使用`pip install leaves`指令安装包后，你可以按照以下步骤获得加速。

列表 6.19 `lleaves`加速 LightGBM 预测
