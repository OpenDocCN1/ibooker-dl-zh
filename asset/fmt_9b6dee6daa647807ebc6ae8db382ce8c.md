
```py
def rescale_image(image: jax.Array) -> jax.Array:
  """Normalizes pixel values to the [0, 1] range by dividing by 255."""
  return image / 255.0
```

现在让我们应用所有预处理步骤——读取 JPEG 图像、保持宽高比调整大小、中心裁剪和归一化——以确保在训练之前我们的数据集是标准化的。虽然我们可以选择任何图像大小，但我们将使用 224 × 224 像素来匹配原始 ResNet 架构期望的输入大小，我们将在本章后面探讨这一点。由于这些转换只需要应用一次，提前进行可以提高效率并避免在训练期间进行冗余计算。

在图 5-10 中，你可以看到预处理前后图像的示例：

```py
original_image = Image.open(metadata.iloc[0]["full_path"])
image = np.array(original_image)
image = resize_preserve_aspect(image)
image = center_crop(image)
image = rescale_image(image)

_, axes = plt.subplots(1, 2, figsize=(6, 3))
axes = axes.flatten()
axes[0].imshow(original_image)
axes[0].set_title("Original")
axes[1].imshow(image)
axes[1].set_title("Preprocessed");
```

![](img/dlfb_0510.png)

###### 图 5-10。图像被标准化以确保一致的尺寸和像素值范围。请注意，尽管预处理过的图像已经从 0-255 缩放到 0-1，但`imshow`会自动调整显示比例，因此视觉外观保持不变。

在完成数据预处理主题后，我们现在可以专注于高效地存储和访问数据——这是在不超载系统内存的情况下扩展训练的关键步骤。

## 使用内存映射数组进行数据存储

我们已经强调了模型训练中效率的重要性：更快的数据处理意味着更快的迭代，最终得到更好的模型。我们想要避免的一个主要低效问题是每次将批次传递给模型时都重复预处理相同的数据。同时，将所有处理过的图像存储在内存中也是不可行的。那么，我们如何平衡效率和内存限制呢？

一个实用的解决方案是*内存映射 NumPy 数组*。这允许我们将预处理过的图像存储在磁盘上，同时像它们在内存中一样访问它们，使检索快速高效。这样，我们在训练前一次性预处理所有图像，然后在训练期间动态加载它们，而不进行冗余计算。

实现很简单：我们首先定义存储文件，指定数据类型（`float32`），并设置数据集的形状：224 × 224 像素，三个颜色通道。这将在磁盘上创建一个空文件，预分配所需的存储空间。然后我们在内存映射数组中预处理和存储每个图像：

```py
from tempfile import TemporaryFile

images_on_disk = np.memmap(
  TemporaryFile(), dtype="float32", mode="w+", shape=(10, 224, 224, 3)
)
for i, full_path in enumerate(metadata[:10]["full_path"]):
  image = Image.open(full_path)
  image = np.array(image)
  image = resize_preserve_aspect(image)
  image = center_crop(image)
  image = rescale_image(image)
  images_on_disk[i, :, :, :] = image
images_on_disk.flush()
```

这种方法让我们得到了两全其美的效果：高效的预处理而不会过度消耗内存。

## 构建一个`DatasetBuilder`

到目前为止，我们已经解决了使用这个数据集的主要挑战：将其分为训练集、验证集和测试集；处理类别不平衡；预处理图像；以及数据增强。现在，我们将这些组件组织成一个结构化的`DatasetBuilder`类，该类返回具有数据迭代器的`Dataset`实例，这些迭代器将实际为我们获取可以用于训练或评估模型的批次数据。

###### 小贴士

设计一个灵活的数据集构建器在迭代模型时会有回报。它让你可以快速尝试不同的标签映射、采样策略和预处理步骤——而无需重写核心代码。这种模块化加快了实验速度，并帮助你确定哪些数据选择实际上提高了性能。

让我们更详细地看看`DatasetBuilder`，它协调数据集分割和预处理的创建：

```py
class DatasetBuilder:
  """Builds a dataset with metadata, loaded images, and class mappings."""

  def __init__(self, data_dir: str, out_dir: str | None = None) -> None:
    self.data_dir = data_dir
    self.out_dir = out_dir or data_dir

  def build(
    self,
    rng: jax.Array,
    splits: dict[str, float],
    preprocessors: list[Callable] = [crop],
    image_size: tuple[int, int, int] = (224, 224, 3),
    class_map: dict[str, Any] | None = None,
  ) -> dict[str, Dataset]:
    """Builds the dataset splits from loaded metadata and loaded images."""
    metadata = MetadataLoader(self.data_dir, self.out_dir).load(class_map)
    images = ImageLoader(metadata, self.out_dir).load(preprocessors, image_size)

    # Shuffle the dataset and assign each example to one of the dataset splits.
    num_samples = len(metadata)
    rng, rng_perm = jax.random.split(rng, 2)
    shuffled_indices = jax.random.permutation(rng_perm, num_samples)

    # Create each dataset split using the shuffled indices and store the
    # corresponding metadata and image data in a Dataset object.
    dataset_splits, start = {}, 0
    for name, size in self._get_split_sizes(splits, num_samples):
      indices = np.array(shuffled_indices[start : (start + size)])
      dataset_splits[name] = Dataset(
        metadata=metadata.iloc[indices],
        images=images,
        num_classes=metadata["class"].nunique(),
      )
      start += size
    return dataset_splits

  def _get_split_sizes(self, splits: dict[str, float], num_samples: int):
    """Convert fractional split sizes to integer counts that sum to total."""
    names = list(splits)
    sizes = [int(num_samples * splits[name]) for name in names[:-1]]
    sizes.append(num_samples - sum(sizes))  # last split gets the remainder
    yield from zip(names, sizes)
```

我们已经实现了这些组件中的每一个，但让我们强调一些关键细节：

**元数据加载**

`MetadataLoader` 确保所有图像文件路径及其相关的类/分割信息都一致地保存到 CSV 文件中。这避免了依赖于`rglob`导致的不一致性，因为`rglob`不能保证文件顺序的一致性。由于图像数据稍后以与元数据相同的顺序存储在内存映射数组中，保持这种一致性至关重要。

**灵活的类映射**

`MetadataLoader` 通过可定制的`class_map`支持二进制或分组分类设置。如果没有提供映射，它将默认为基于目录结构的标准多类配置。

**图像预处理**

`ImageLoader` 负责在内存映射数组中读取、处理和存储图像。为了避免重复计算，图像一旦存储为紧凑的原始格式，就可以应用多种预处理策略（例如，调整大小或裁剪），并将它们分别作为 memmaps 存储起来。

**数据集创建**

`build`方法将所有内容整合在一起：它加载元数据和图像，对数据集进行洗牌，将其分割为训练/验证/测试分区，并将每个分区包装成一个`Dataset`对象。每个数据集包含一致的元数据和预处理的图像视图，准备好进行训练。

### 加载元数据和图像

你可以在`dlfb.cancer.dataset.builder`库中找到`MetadataLoader`和`ImageLoader`的支持类代码。在这里，我们将简要描述它们的工作原理以及它们如何履行创建`Dataset`的职责。

如果不存在，`MetadataLoader`将构建一个元数据 CSV 文件。它从文件夹结构中提取原始分割和类信息，应用任何所需的标签映射，并提供对数据集的行级访问。图像加载稍微复杂一些。虽然`MetadataLoader`处理文件路径和类标签，但`ImageLoader`负责读取、预处理和存储实际的图像数据。以下是关于`ImageLoader`的工作原理的一些说明：

**图像预处理和存储**

为了避免冗余计算，`ImageLoader`首先将原始图像存储为一个单一的内存映射数组。这个数组紧凑，存储了每个图像的展平版本以及它们的原始形状和偏移量，以便稍后可以重建。一旦完成这个原始存储，就可以应用各种预处理函数（如裁剪或调整大小），并将结果保存到额外的内存映射文件中。这种设计使我们能够支持多种预处理策略，而无需重新加载原始的 JPEG 图像或重复昂贵的操作。

**预处理灵活性**

预处理函数以模块化的方式应用。可以将任意数量的转换传递给`ImageLoader`，每个转换都会生成自己的内存映射。这些预处理的视图存储在`Images`对象的不同键下（例如，`"crop"`，`"resize"`），该对象作为统一的接口来访问它们。

**数据集组装**

最后，`DatasetBuilder.build()`方法将这些组件整合在一起。它加载元数据，处理图像，对数据集进行洗牌，将其分为训练、验证和测试集，并为每个返回`Dataset`对象。这些对象提供了对图像数组和类标签的干净访问，为模型训练做好准备。

所有这些都存储在`Dataset`类本身中：

```py
@dataclass
class Dataset:
  """Dataset class storing images and corresponding metadata."""

  metadata: pd.DataFrame
  images: Images
  num_classes: int

  def get_dummy_input(self) -> jnp.ndarray:
    """Returns dummy input with the correct shape for the model."""
    return jnp.empty((1,) + self.images.size)

  def num_samples(self) -> int:
    """Returns the number of samples in the dataset."""
    return self.metadata.shape[0]

  def get_images(self, preprocessor: Callable, indices: jax.Array) -> jax.Array:
    """Returns preprocessed images for the given indices."""
    return self.images.loaded[preprocessor.__name__][indices]

  def get_labels(self, indices: list[int]) -> jax.Array:
    """Returns integer class labels for the given indices."""
    return jnp.int16(self.metadata.loc[indices]["label"].values)
```

`Images`在哪里：

```py
@dataclass
class Images:
  """Stores image data and size information."""

  loaded: dict[str, np.memmap]
  size: tuple[int, int, int]
```

`Dataset`类是训练期间访问数据的初级接口。它管理`metadata`（跟踪图像路径和标签）和内存映射的`images`。由于我们将数据集分为训练、验证和测试集，我们可以使用`splits`参数控制它们的比例。

### 数据批处理

为了简化数据集处理，在训练期间，我们引入了`BatchHandler`类。这个类封装了从`Dataset`对象生成批次的所需组件：

- `preprocessor`选择要使用的预处理图像视图（例如，裁剪、调整大小）。
- `sampler`确定如何为每个批次抽取样本（例如，平衡抽样与随机抽样）。
- 可选的`augmentor`应用实时图像变换，如翻转、颜色抖动或旋转，以提高泛化能力。
