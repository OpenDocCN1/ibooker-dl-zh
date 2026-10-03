
```py
local bert_model = "bert-base-cased";

{
    "dataset_reader": {
        "type": "snli",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
            "add_special_tokens": false
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        }
    },
    "train_data_path": "https://realworldnlpbook.s3.amazonaws.com/data/snli/snli_1.0_train.jsonl",
    "validation_data_path": "https://realworldnlpbook.s3.amazonaws.com/data/snli/snli_1.0_dev.jsonl",

    "model": {
        "type": "basic_classifier",

        "text_field_embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "seq2vec_encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["tokens"],
            "padding_noise": 0.1,
            "batch_size" : 32
        }
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5.0e-6
        },
        "validation_metric": "+accuracy",
        "num_epochs": 30,
        "patience": 10,
        "cuda_device": 0
    }
}
```

如果您不熟悉数据加载器和训练器部分正在发生的事情也没关系。我们将在第十章讨论这些主题（批处理、填充、优化、超参数调整）。在将此配置文件保存在`examples/nli/snli_transformers.jsonnet`后，您可以通过运行以下代码开始训练过程：

```py
allennlp train examples/nli/snli_transformers.jsonnet --serialization-dir models/snli
```

这将运行一段时间（即使在诸如 Nvidia V100 这样的快速 GPU 上也是如此），并在 `stdout` 上产生大量的日志消息。以下是我在四个时期后得到的日志消息的片段：

```py
...
allennlp.training.trainer - Epoch 4/29
allennlp.training.trainer - Worker 0 memory usage MB: 6644.208
allennlp.training.trainer - GPU 0 memory usage MB: 8708
allennlp.training.trainer - Training
allennlp.training.trainer - Validating
allennlp.training.tensorboard_writer -                        Training |  Validation
allennlp.training.tensorboard_writer - accuracy           |     0.933  |     0.908
allennlp.training.tensorboard_writer - gpu_0_memory_MB    |  8708.000  |       N/A
allennlp.training.tensorboard_writer - loss               |     0.190  |     0.293
allennlp.training.tensorboard_writer - reg_loss           |     0.000  |     0.000
allennlp.training.tensorboard_writer - worker_0_memory_MB |  6644.208  |       N/A
allennlp.training.checkpointer - Best validation performance so far. Copying weights to 'models/snli/best.th'.
allennlp.training.trainer - Epoch duration: 0:21:39.687226
allennlp.training.trainer - Estimated training time remaining: 9:04:56
...
```

注意验证准确率（`0.908`）。考虑到这是一个三类分类，随机基线只会是 `0.3`。相比之下，当我用基于 LSTM 的 RNN 替换 BERT 时，我得到的最佳验证准确率约为 `~0.68`。我们需要更仔细地进行实验，以公平地比较不同模型之间的差异，但这个结果似乎表明 BERT 是解决自然语言理解问题的强大模型。

## 摘要

+   转移学习是一个机器学习概念，其中一个模型学习了一个任务，然后通过在它们之间转移知识来应用到另一个任务上。这是许多现代、强大、预训练模型的基本概念。

+   BERT 是一个使用掩码语言建模和下一句预测目标进行预训练的 Transformer 编码器，以产生上下文化的嵌入，一系列考虑上下文的词嵌入。

+   ELMo、XLNet、RoBERTa、DistilBERT 和 ALBERT 是现代深度自然语言处理中常用的其他流行的预训练模型。

+   你可以直接使用 Hugging Face 的 Transformers 库构建基于 BERT 的 NLP 应用，也可以使用无缝集成 Transformers 库的 AllenNLP。

^(1.)Ramponi 和 Plank，“NLP 中的神经无监督领域自适应——一项调查”，（2020）。[`arxiv.org/abs/2006.00632`](https://arxiv.org/abs/2006.00632)。

^(2.)Jacob Devlin，Ming-Wei Chang，Kenton Lee 和 Kristina Toutanova，“BERT：用于语言理解的深度双向 Transformer 预训练”，（2018）。[`arxiv.org/abs/1810.04805`](https://arxiv.org/abs/1810.04805)。

^(3.)Bryan McCann，James Bradbury，Caiming Xiong 和 Richard Socher，“翻译中学到的：上下文化的词向量”，2017 年 NIPS 会议。

^(4.)Peters 等人，“深度上下文化的词表示”，（2018）。[`arxiv.org/abs/1802.05365`](https://arxiv.org/abs/1802.05365)。

^(5.)点击这里查看有关如何使用 ELMo 与 AllenNLP 的详细文档：[`allennlp.org/elmo`](https://allennlp.org/elmo)。

^(6.)请参阅[`huggingface.co/transformers/model_doc/xlnet.html`](https://huggingface.co/transformers/model_doc/xlnet.html)以获取文档。

^(7.)Liu 等人，“RoBERTa：一个稳健优化的 BERT 预训练方法，”（2019）。[`arxiv.org/abs/1907.11692`](https://arxiv.org/abs/1907.11692)。

^(8.)Conneau 等人，“规模化的无监督跨语言表示学习”，（2019）。[`arxiv.org/abs/1911.02116`](https://arxiv.org/abs/1911.02116)。

^(9.)Sanh 等人，“DistilBERT，BERT 的精简版：更小、更快、更便宜、更轻”，（2019）。[`arxiv.org/abs/1910.01108`](https://arxiv.org/abs/1910.01108)。

^(10.)Lan 等人，“ALBERT：一种用于自监督学习语言表示的轻量级 BERT”，（2020）。[`arxiv.org/abs/1909.11942`](https://arxiv.org/abs/1909.11942)。

^(11.)这些示例来自 ROCStories：[`cs.rochester.edu/nlp/rocstories/`](https://cs.rochester.edu/nlp/rocstories/)。

^(12.)这些示例来自于[`nlpprogress.com/english/natural_language_inference.html`](http://nlpprogress.com/english/natural_language_inference.html).

^(13.)Reimers 和 Gurevych，“使用 Siamese BERT-Network 的句子嵌入：Sentence-BERT”，(2019)。[`arxiv.org/abs/1908.10084`](https://arxiv.org/abs/1908.10084).

^(14.)Seo 等人，“面向机器理解的双向注意力流”，(2018)。[`arxiv.org/abs/1611.01603`](https://arxiv.org/abs/1611.01603).
