
这个 protobuf 定义表示我们正在使用 protobuf 格式的第 3 版，并且指定每个`Person`对象（可选）可能具有一个字符串类型的`name`、一个 int32 类型的`id`，以及零个或多个字符串类型的`email`字段。数字`1`、`2`和`3`是字段标识符：它们将在每个记录的二进制表示中使用。一旦你在`*.proto*`文件中有了一个定义，你就可以编译它。这需要使用 protobuf 编译器`protoc`在 Python（或其他语言）中生成访问类。请注意，你通常在 TensorFlow 中使用的 protobuf 定义已经为你编译好了，并且它们的 Python 类是 TensorFlow 库的一部分，因此你不需要使用`protoc`。你只需要知道如何在 Python 中*使用*protobuf 访问类。为了说明基础知识，让我们看一个简单的示例，使用为`Person`protobuf 生成的访问类（代码在注释中有解释）：

```py
>>> from person_pb2 import Person  # import the generated access class
>>> person = Person(name="Al", id=123, email=["a@b.com"])  # create a Person
>>> print(person)  # display the Person
name: "Al"
id: 123
email: "a@b.com"
>>> person.name  # read a field
'Al'
>>> person.name = "Alice"  # modify a field
>>> person.email[0]  # repeated fields can be accessed like arrays
'a@b.com'
>>> person.email.append("c@d.com")  # add an email address
>>> serialized = person.SerializeToString()  # serialize person to a byte string
>>> serialized
b'\n\x05Alice\x10{\x1a\x07a@b.com\x1a\x07c@d.com'
>>> person2 = Person()  # create a new Person
>>> person2.ParseFromString(serialized)  # parse the byte string (27 bytes long)
27
>>> person == person2  # now they are equal
True
```

简而言之，我们导入由`protoc`生成的`Person`类，创建一个实例并对其进行操作，可视化它并读取和写入一些字段，然后使用`SerializeToString()`方法对其进行序列化。这是准备保存或通过网络传输的二进制数据。当读取或接收这些二进制数据时，我们可以使用`ParseFromString()`方法进行解析，并获得被序列化的对象的副本。

你可以将序列化的`Person`对象保存到 TFRecord 文件中，然后加载和解析它：一切都会正常工作。然而，`ParseFromString()`不是一个 TensorFlow 操作，所以你不能在 tf.data 管道中的预处理函数中使用它（除非将其包装在`tf.py_function()`操作中，这会使代码变慢且不太可移植，正如你在第十二章中看到的）。然而，你可以使用`tf.io.decode_proto()`函数，它可以解析任何你想要的 protobuf，只要你提供 protobuf 定义（请参考笔记本中的示例）。也就是说，在实践中，你通常会希望使用 TensorFlow 提供的专用解析操作的预定义 protobuf。现在让我们来看看这些预定义的 protobuf。

## TensorFlow Protobufs

TFRecord 文件中通常使用的主要 protobuf 是`Example`protobuf，它表示数据集中的一个实例。它包含一个命名特征列表，其中每个特征可以是一个字节字符串列表、一个浮点数列表或一个整数列表。以下是 protobuf 定义（来自 TensorFlow 源代码）：

```py
syntax = "proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
```

`BytesList`、`FloatList`和`Int64List`的定义足够简单明了。请注意，对于重复的数值字段，使用`[packed = true]`进行更有效的编码。`Feature`包含一个`BytesList`、一个`FloatList`或一个`Int64List`。一个`Features`（带有`s`）包含一个将特征名称映射到相应特征值的字典。最后，一个`Example`只包含一个`Features`对象。

###### 注意

为什么会定义`Example`，因为它只包含一个`Features`对象？嗯，TensorFlow 的开发人员可能有一天决定向其中添加更多字段。只要新的`Example`定义仍然包含相同 ID 的`features`字段，它就是向后兼容的。这种可扩展性是 protobuf 的一个伟大特性。

这是你如何创建一个代表同一个人的`tf.train.Example`：

```py
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
        }))
```

这段代码有点冗长和重复，但你可以很容易地将其包装在一个小的辅助函数中。现在我们有了一个`Example` protobuf，我们可以通过调用其`SerializeToString()`方法将其序列化，然后将生成的数据写入 TFRecord 文件。让我们假装写入五次，以假装我们有几个联系人：

```py
with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    for _ in range(5):
        f.write(person_example.SerializeToString())
```

通常，您会写比五个`Example`更多的内容！通常情况下，您会创建一个转换脚本，从当前格式（比如 CSV 文件）读取数据，为每个实例创建一个`Example` protobuf，将它们序列化，并保存到几个 TFRecord 文件中，最好在此过程中对它们进行洗牌。这需要一些工作，所以再次确保这确实是必要的（也许您的流水线使用 CSV 文件运行良好）。

现在我们有一个包含多个序列化`Example`的漂亮 TFRecord 文件，让我们尝试加载它。

## 加载和解析示例

为了加载序列化的`Example` protobufs，我们将再次使用`tf.data.TFRecordDataset`，并使用`tf.io.parse_single_example()`解析每个`Example`。它至少需要两个参数：包含序列化数据的字符串标量张量，以及每个特征的描述。描述是一个字典，将每个特征名称映射到`tf.io.FixedLenFeature`描述符，指示特征的形状、类型和默认值，或者`tf.io.VarLenFeature`描述符，仅指示特征列表的长度可能变化的类型（例如`"emails"`特征）。

以下代码定义了一个描述字典，然后创建了一个`TFRecordDataset`，并对其应用了一个自定义预处理函数，以解析该数据集包含的每个序列化`Example` protobuf：

```py
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

def parse(serialized_example):
    return tf.io.parse_single_example(serialized_example, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).map(parse)
for parsed_example in dataset:
    print(parsed_example)
```

固定长度的特征被解析为常规张量，但变长特征被解析为稀疏张量。您可以使用`tf.sparse.to_dense()`将稀疏张量转换为密集张量，但在这种情况下，更简单的方法是直接访问其值：

```py
>>> tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
>>> parsed_example["emails"].values
<tf.Tensor: [...] dtype=string, numpy=array([b'a@b.com', b'c@d.com'], [...])>
```

您可以使用`tf.io.parse_example()`批量解析示例，而不是使用`tf.io.parse_single_example()`逐个解析它们：

```py
def parse(serialized_examples):
    return tf.io.parse_example(serialized_examples, feature_description)

dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(2).map(parse)
for parsed_examples in dataset:
    print(parsed_examples)  # two examples at a time
```

最后，`BytesList`可以包含您想要的任何二进制数据，包括任何序列化对象。例如，您可以使用`tf.io.encode_jpeg()`使用 JPEG 格式对图像进行编码，并将这些二进制数据放入`BytesList`中。稍后，当您的代码读取 TFRecord 时，它将从解析`Example`开始，然后需要调用`tf.io.decode_jpeg()`来解析数据并获取原始图像（或者您可以使用`tf.io.decode_image()`，它可以解码任何 BMP、GIF、JPEG 或 PNG 图像）。您还可以通过使用`tf.io.serialize_tensor()`对张量进行序列化，然后将生成的字节字符串放入`BytesList`特征中，将任何您想要的张量存储在`BytesList`中。稍后，当您解析 TFRecord 时，您可以使用`tf.io.parse_tensor()`解析这些数据。请参阅本章的笔记本[*https://homl.info/colab3*](https://homl.info/colab3) ，了解在 TFRecord 文件中存储图像和张量的示例。

正如您所看到的，`Example` protobuf 非常灵活，因此对于大多数用例来说可能已经足够了。但是，当您处理列表列表时，可能会有些繁琐。例如，假设您想对文本文档进行分类。每个文档可以表示为一个句子列表，其中每个句子表示为一个单词列表。也许每个文档还有一个评论列表，其中每个评论表示为一个单词列表。还可能有一些上下文数据，比如文档的作者、标题和发布日期。TensorFlow 的`SequenceExample` protobuf 就是为这种用例而设计的。

## 使用 SequenceExample Protobuf 处理列表列表

这是`SequenceExample` protobuf 的定义：

```py
message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
    Features context = 1;
    FeatureLists feature_lists = 2;
};
```

`SequenceExample`包含一个`Features`对象用于上下文数据和一个包含一个或多个命名`FeatureList`对象（例如，一个名为`"content"`的`FeatureList`和另一个名为`"comments"`的`FeatureList`）的`FeatureLists`对象。每个`FeatureList`包含一个`Feature`对象列表，每个`Feature`对象可能是字节字符串列表、64 位整数列表或浮点数列表（在此示例中，每个`Feature`可能代表一个句子或评论，可能以单词标识符列表的形式）。构建`SequenceExample`、序列化它并解析它类似于构建、序列化和解析`Example`，但您必须使用`tf.io.parse_single_sequence_example()`来解析单个`SequenceExample`或`tf.io.parse_sequence_example()`来解析批处理。这两个函数返回一个包含上下文特征（作为字典）和特征列表（也作为字典）的元组。如果特征列表包含不同大小的序列（如前面的示例），您可能希望使用`tf.RaggedTensor.from_sparse()`将它们转换为不规则张量（请参阅完整代码的笔记本）：

```py
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```

现在您已经知道如何使用 tf.data API、TFRecords 和 protobufs 高效存储、加载、解析和预处理数据，是时候将注意力转向 Keras 预处理层了。

# Keras 预处理层

为神经网络准备数据通常需要对数值特征进行归一化、对分类特征和文本进行编码、裁剪和调整图像等。有几种选项：

+   预处理可以提前在准备训练数据文件时完成，使用您喜欢的任何工具，如 NumPy、Pandas 或 Scikit-Learn。您需要在生产中应用完全相同的预处理步骤，以确保您的生产模型接收到与训练时相似的预处理输入。

+   或者，您可以在加载数据时使用 tf.data 进行即时预处理，通过使用该数据集的`map()`方法对数据集的每个元素应用预处理函数，就像本章前面所做的那样。同样，您需要在生产中应用相同的预处理步骤。

+   最后一种方法是直接在模型内部包含预处理层，这样它可以在训练期间即时预处理所有输入数据，然后在生产中使用相同的预处理层。本章的其余部分将讨论这种最后一种方法。

Keras 提供了许多预处理层，您可以将其包含在模型中：它们可以应用于数值特征、分类特征、图像和文本。我们将在接下来的部分中讨论数值和分类特征，以及基本文本预处理，我们将在第十四章中涵盖图像预处理，以及在第十六章中涵盖更高级的文本预处理。

## 归一化层

正如我们在第十章中看到的，Keras 提供了一个`Normalization`层，我们可以用来标准化输入特征。我们可以在创建层时指定每个特征的均值和方差，或者更简单地在拟合模型之前将训练集传递给该层的`adapt()`方法，以便该层可以在训练之前自行测量特征的均值和方差。
