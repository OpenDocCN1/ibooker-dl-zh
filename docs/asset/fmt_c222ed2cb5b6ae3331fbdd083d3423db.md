
# 神经机器翻译模型

```py
class NMTModel(nn.Module):
    """ A Neural Machine Translation Model """
    def __init__(self, source_vocab_size, source_embedding_size,
                 target_vocab_size, target_embedding_size, encoding_size,
                 target_bos_index):
        """
        Args:
            source_vocab_size (int): number of unique words in source language
            source_embedding_size (int): size of the source embedding vectors
            target_vocab_size (int): number of unique words in target language
            target_embedding_size (int): size of the target embedding vectors
            encoding_size (int): the size of the encoder RNN.
            target_bos_index (int): index for BEGIN-OF-SEQUENCE token
        """
        super(NMTModel, self).__init__()
        self.encoder = NMTEncoder(num_embeddings=source_vocab_size,
                                  embedding_size=source_embedding_size,
                                  rnn_hidden_size=encoding_size)
        decoding_size = encoding_size * 2
        self.decoder = NMTDecoder(num_embeddings=target_vocab_size,
                                  embedding_size=target_embedding_size,
                                  rnn_hidden_size=decoding_size,
                                  bos_index=target_bos_index)

    def forward(self, x_source, x_source_lengths, target_sequence):
        """The forward pass of the model

        Args:
            x_source (torch.Tensor): the source text data tensor.
                x_source.shape should be (batch, vectorizer.max_source_length)
            x_source_lengths torch.Tensor): the length of the sequences in x_source
            target_sequence (torch.Tensor): the target text data tensor
        Returns:
            decoded_states (torch.Tensor): prediction vectors at each output step
        """
        encoder_state, final_hidden_states = self.encoder(x_source,
                                                          x_source_lengths)
        decoded_states = self.decoder(encoder_state=encoder_state,
                                      initial_hidden_state=final_hidden_states,
                                      target_sequence=target_sequence)
        return decoded_states
```

## 编码器

示例 8-5：编码器嵌入了源单词和使用 bi-GPU 提取的特征

```py
class NMTEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size):
        """
        Args:
            num_embeddings (int): size of source vocabulary
            embedding_size (int): size of the embedding vectors
            rnn_hidden_size (int): size of the RNN hidden state vectors
        """
        super(NMTEncoder, self).__init__()

        self.source_embedding = nn.Embedding(num_embeddings, embedding_size,
                                             padding_idx=0)
        self.birnn = nn.GRU(embedding_size, rnn_hidden_size, bidirectional=True,
                            batch_first=True)

    def forward(self, x_source, x_lengths):
        """The forward pass of the model

        Args:
            x_source (torch.Tensor): the input data tensor.
                x_source.shape is (batch, seq_size)
            x_lengths (torch.Tensor): vector of lengths for each item in the batch
        Returns:
            a tuple: x_unpacked (torch.Tensor), x_birnn_h (torch.Tensor)
                x_unpacked.shape = (batch, seq_size, rnn_hidden_size * 2)
                x_birnn_h.shape = (batch, rnn_hidden_size * 2)
        """
        x_embedded = self.source_embedding(x_source)
        # create PackedSequence; x_packed.data.shape=(number_items, embedding_size)
        x_lengths = x_lengths.detach().cpu().numpy()
        x_packed = pack_padded_sequence(x_embedded, x_lengths, batch_first=True)

        # x_birnn_h.shape = (num_rnn, batch_size, feature_size)
        x_birnn_out, x_birnn_h  = self.birnn(x_packed)
        # permute to (batch_size, num_rnn, feature_size)
        x_birnn_h = x_birnn_h.permute(1, 0, 2)

        # flatten features; reshape to (batch_size, num_rnn * feature_size)
        #  (recall: -1 takes the remaining positions,
        #           flattening the two RNN hidden vectors into 1)
        x_birnn_h = x_birnn_h.contiguous().view(x_birnn_h.size(0), -1)

        x_unpacked, _ = pad_packed_sequence(x_birnn_out, batch_first=True)
        return x_unpacked, x_birnn_h
```

通常，编码器将整数序列作为输入，并为每个位置创建特征向量。在该示例中，编码器的输出是这些向量以及用于制作特征向量的 bi-GRU 的最终隐藏状态。该隐藏状态用于在下一节中初始化解码器的隐藏状态。

深入了解编码器，我们首先使用嵌入层嵌入输入序列。通常，只需在嵌入层上设置`padding_idx`标志，我们就可以使模型处理可变长度序列，因为任何等于`padding_idx`的位置都会被赋予零值向量，该向量在优化期间不会更新。回想一下，这被称为面具。然而，在这种编码器-解码器模型中，掩蔽位置需要以不同方式处理，因为我们使用 bi-GRU 来编码源序列。主要原因是后向分量可能受到屏蔽位置的影响，其因子与在序列上开始之前遇到的屏蔽位置的数量成比例。

为了处理 bi-GRU 中可变长度序列的掩码位置，我们使用 PyTorch 的`PackedSequence`数据结构。`PackedSequences`源自 CUDA 如何允许以批量格式处理可变长度序列。如果满足两个条件，则可以将任何零填充序列（例如示例 8-6 中所示的编码器中的嵌入源序列）转换为`PackedSequence`：提供每个序列的长度，并根据以下顺序对小批量进行排序。这些序列的长度。这在图 8-11 中以可视方式显示，因为它是一个复杂的主题，我们在例 8-6 及其输出中再次演示它。

### 示例 8-6：`packed_padded_sequences`和`pad_packed_sequences`的简单演示

```py
Input[0]
abcd_padded = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
efg_padded = torch.tensor([5, 6, 7, 0], dtype=torch.float32)
h_padded = torch.tensor([8, 0, 0, 0], dtype=torch.float32)

padded_tensor = torch.stack([abcd_padded, efg_padded, h_padded])

describe(padded_tensor)
Output[0]
Type: torch.FloatTensor
Shape/size: torch.Size([3, 4])
Values:
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  0.],
        [ 8.,  0.,  0.,  0.]])
Input[1]
lengths = [4, 3, 1]
packed_tensor = pack_padded_sequence(padded_tensor, lengths,   
                                     batch_first=True)
packed_tensor
Output[1]
PackedSequence(data=tensor([ 1.,  5.,  8.,  2.,  6.,  3.,  7.,  4.]),
               batch_sizes=tensor([ 3,  2,  2,  1]))
Input[2]
unpacked_tensor, unpacked_lengths = \
    pad_packed_sequence(packed_tensor, batch_first=True)

describe(unpacked_tensor)
describe(unpacked_lengths)
Output[2]
Type: torch.FloatTensor
Shape/size: torch.Size([3, 4])
Values:
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  0.],
        [ 8.,  0.,  0.,  0.]])
Type: torch.LongTensor
Shape/size: torch.Size([3])
Values:
tensor([ 4,  3,  1])
```

我们在生成每个小批量时处理排序，如上一节所述。然后，如例 8-7 所示，通过传递嵌入的序列，序列的长度和表示第一个维度是批量维度的布尔标志来激发 PyTorch 的`pack_padded_sequence`函数。此函数的输出是`PackedSequence`。将得到的`PackedSequence`输入到 bi-GRU 中以为下游解码器创建状态向量。使用另一个布尔标志将 bi-GRU 的输出解压缩为完整张量，指示批量在第一维上。解包操作，如图 8-11 所示，将每个屏蔽位置 15 设置为零值向量，保留下游计算的完整性。

### 示例 8-7：NMT 解码器从编码的源句子中构造目标句子
