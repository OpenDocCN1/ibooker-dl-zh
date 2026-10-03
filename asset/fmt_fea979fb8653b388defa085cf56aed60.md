

# 第九章：重用预训练的图像识别网络

图像识别和计算机视觉是深度学习取得重大影响的领域之一。拥有几十层甚至超过一百层的网络已经被证明在图像分类任务中非常有效，甚至超过了人类。

然而，训练这样的网络非常复杂，无论是在处理能力还是所需的训练图像数量方面。幸运的是，我们通常不必从头开始，而是可以重用现有的网络。

在本章中，我们将介绍如何加载 Keras 支持的五个预训练网络之一，讨论在将图像输入网络之前需要的预处理，最后展示如何在推理模式下运行网络，询问它认为图像包含什么。

接下来，我们将探讨所谓的*迁移学习*——将一个预训练网络部分重新训练以适应新任务的新数据。我们首先从 Flickr 获取一组包含猫和狗的图像。然后教会我们的网络区分它们。接下来，我们将应用这个网络来改进 Flickr 的搜索结果。最后，我们将下载一组包含 37 种不同宠物图片的图像，并训练一个网络，使其在标记它们方面超过平均人类。

以下笔记本包含本章中提到的代码：

```py
09.1 Reusing a pretrained image recognition network
09.2 Images as embeddings
09.3 Retraining
```

## 9.1 加载预训练网络

### 问题

您想知道如何实例化一个预训练的图像识别网络。

### 解决方案

使用 Keras 加载一个预训练网络，如果需要的话下载权重。

Keras 不仅使得组合网络更容易，还提供了对各种预训练网络的引用，我们可以轻松加载：

```py
model = VGG16(weights='imagenet', include_top=True)
model.summary()
```

这也将打印网络的摘要，显示其各个层。当我们想要使用网络时，这是有用的，因为它不仅显示层的名称，还显示它们的大小以及它们是如何连接的。

### 讨论

Keras 提供了访问多个流行的图像识别网络的功能，可以直接下载。这些下载被缓存在`~/.keras/models/`中，所以通常您只需要在第一次下载时等待。

总共我们可以使用五种不同的网络（`VGG16`、`VGG19`、`ResNet50`、`Inception V3`和`Xception`）。它们在复杂性和架构上有所不同，但对于大多数简单的应用程序来说，可能不太重要选择哪个模型。`VGG16`只有 16 层深度，更容易检查。`Inception`是一个更深的网络，但变量少了 85%，这使得加载更快，占用内存更少。

## 9.2 图像预处理

### 问题

您已经加载了一个预训练网络，但现在需要知道如何在将图像输入网络之前对图像进行预处理。

### 解决方案

裁剪和调整图像到正确的大小，并对颜色进行归一化。

Keras 包含的所有预训练网络都期望它们的输入是方形的，并且具有特定的大小。它们还期望颜色通道被归一化。在训练时对图像进行归一化使得网络更容易专注于重要的事物，而不会被“分心”。

我们可以使用 PIL/Pillow 加载和中心裁剪图像：

```py
img = Image.open('data/cat.jpg')
w, h = img.size
s = min(w, h)
y = (h - s) // 2
x = (w - s) // 2
img = img.crop((x, y, s, s))
```

通过查询`input_shape`属性，我们可以从网络的第一层获取所需的大小。该属性还包含颜色深度，但根据架构的不同，这可能是第一个或最后一个维度。通过对其调用`max`，我们将得到正确的数字：

```py
target_size = max(model.layers[0].input_shape)
img = img.resize((target_size, target_size), Image.ANTIALIAS)
imshow(np.asarray(img))
```

![我们猫的处理图像](img/dlcb_09in01.png)

最后，我们需要将图像转换为适合网络处理的格式。这涉及将图像转换为数组，扩展维度使其成为一个批次，并对颜色进行归一化：

```py
np_img = image.img_to_array(img)
img_batch = np.expand_dims(np_img, axis=0)
pre_processed = preprocess_input(img_batch)
pre_processed.shape
```

```py
(1, 224, 224, 3)
```

我们现在准备对图像进行分类！

### 讨论

中心裁剪并不是唯一的选择。事实上，Keras 的`image`模块中有一个名为`load_img`的函数，它可以加载和调整图像的大小，但不进行裁剪。尽管如此，这是一个将图像转换为网络期望大小的良好通用策略。

中心裁剪通常是最佳策略，因为我们想要分类的内容通常位于图像中间，直接调整大小会扭曲图片。但在某些情况下，特殊策略可能效果更好。例如，如果我们有很高的白色背景图像，那么中心裁剪可能会切掉太多实际图像，而调整大小会导致严重扭曲。在这种情况下，更好的解决方案可能是在两侧用白色像素填充图像，使其变成正方形。

## 9.3 在图像上运行推断

### 问题

如果你有一张图像，如何找出它显示的是什么？

### 解决方案

使用预训练网络对图像进行推断。

一旦我们将图像转换为正确的格式，就可以在模型上调用`predict`：

```py
features = model.predict(pre_processed)
features.shape
```

```py
(1, 1000)
```

预测结果以`numpy`数组的形式返回（1, 1,000）—每个批次中的每个图像对应一个包含 1,000 个元素的向量。向量中的每个条目对应一个标签，而条目的值表示图像代表该标签的可能性有多大。

Keras 有方便的`decode_predictions`函数，可以找到得分最高的条目并返回标签和相应的分数：

```py
decode_predictions(features, top=5)
```

以下是上一个配方中图像的结果：

```py
[[(u'n02124075', u'Egyptian_cat', 0.14703247),
  (u'n04040759', u'radiator', 0.12125628),
  (u'n02123045', u'tabby', 0.097638465),
  (u'n03207941', u'dishwasher', 0.047418527),
  (u'n02971356', u'carton', 0.047036409)]]
```

网络认为我们在看一只猫。它猜测是暖气片有点令人惊讶，尽管背景看起来有点像暖气片。

### 讨论

这个网络的最后一层具有 softmax 激活函数。softmax 函数确保所有类别的激活总和等于 1。由于网络在训练时学习的方式，这些激活可以被视为图像匹配类别的可能性。

所有预训练网络都具有一千个可以识别的图像类别。这是因为它们都是为 [ImageNet 竞赛](http://www.image-net.org/challenges/LSVRC/) 进行训练的。这使得比较它们的相对性能变得容易，但除非我们碰巧想要检测这个竞赛中的图像，否则对于实际目的来说并不立即有用。在下一章中，我们将看到如何使用这些预训练网络来对我们自己选择的图像进行分类。

另一个限制是这些类型的网络只返回一个答案，而图像中通常有多个对象。我们将在第十一章中探讨如何解决这个问题。

## 9.4 使用 Flickr API 收集一组带标签的图像

### 问题

如何快速组合一组带标签的图像进行实验？

### 解决方案

使用 Flickr API 的`search`方法。

要使用 Flickr API，您需要一个应用程序密钥，所以请前往 [*https://www.flickr.com/services/apps/create*](https://www.flickr.com/services/apps/create) 注册您的应用程序。一旦您有了密钥和秘钥，您就可以使用`flickrapi`库搜索图像：

```py
flickr = flickrapi.FlickrAPI(FLICKR_KEY, FLICKR_SECRET, format='parsed-json')
res = flickr.photos.search(text='"cat"', per_page='10', sort='relevance')
photos = res['photos']['photo']
```

Flickr 返回的照片默认不包含 URL。但我们可以从记录中组合 URL：

```py
def flickr_url(photo, size=''):
    url = 'http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg'
    if size:
        size = '_' + size
    return url.format(size=size, **photo)
```

`HTML`方法是在笔记本中显示图像的最简单方法：

```py
tags = ['<img src="{}" width="150" style="display:inline"/>'
        .format(flickr_url(photo)) for photo in photos]
HTML(''.join(tags))
```

这应该显示一堆猫的图片。确认我们有不错的图片后，让我们下载一个稍大一点的测试集：

```py
def fetch_photo(dir_name, photo):
    urlretrieve(flickr_url(photo), os.path.join(dir_name, photo['id'] + '.jpg'))

def fetch_image_set(query, dir_name=None, count=250, sort='relevance'):
    res = flickr.photos.search(text='"{}"'.format(query),
                               per_page=count, sort=sort)['photos']['photo']
    dir_name = dir_name or query
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with multiprocessing.Pool() as p:
        p.map(partial(fetch_photo, dir_name), res)

fetch_image_set('cat')
```

### 讨论

在深度学习实验中运行时，获取良好的训练数据始终是一个关键问题。在图像方面，很难超越 Flickr API，它为我们提供了数十亿张图像。我们不仅可以根据关键字和标签找到图像，还可以根据它们的拍摄地点进行筛选。我们还可以根据如何使用这些图像进行过滤。对于随机实验来说，这并不是一个因素，但如果我们想以某种方式重新发布这些图像，这肯定会派上用场。

Flickr API 让我们可以访问一般的用户生成的图像。根据您的目的，可能有其他可用的 API 更适合。在 Chapter 10 中，我们将看看如何直接从维基百科获取图像。[Getty Images](http://developers.gettyimages.com/) 提供了一个用于库存图像的良好 API，而 [500px](https://github.com/500px/api-documentation) 通过其 API 提供了高质量图像的访问。最后两个对于再发布有严格的要求，但对于实验来说非常好。

## 9.5 构建一个可以区分猫和狗的分类器

### 问题

您希望能够将图像分类为两个类别之一。

### 解决方案

在预训练网络的特征之上训练支持向量机。

让我们从获取狗的训练集开始。

```py
fetch_image_set('dog')
```

将图像加载为一个向量，先是猫，然后是狗：

```py
images = [image.load_img(p, target_size=(224, 224))
          for p in glob('cat/*jpg') + glob('dog/*jpg')]
vector = np.asarray([image.img_to_array(img) for img in images])
```

现在加载预训练模型，并构建一个以`fc2`为输出的新模型。`fc2`是网络分配标签之前的最后一个全连接层。这一层的值描述了图像的抽象方式。另一种说法是，这将图像投影到高维语义空间中：

```py
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('fc2').output)
```

现在我们将在所有图像上运行模型：

```py
vectors = model.predict(vector)
vectors.shape
```

对于我们的 500 张图像中的每一张，我们现在有一个描述该图像的 4,096 维向量。就像在 Chapter 4 中一样，我们可以构建一个支持向量机来找到这个空间中猫和狗之间的区别。

让我们运行支持向量机并打印我们的性能：

```py
X_train, X_test, y_train, y_test = train_test_split(
    p, [1] * 250 + [0] * 250, test_size=0.20, random_state=42)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
sum(1 for p, t in zip(clf.predict(X_test), y_test) if p != t)
```

根据我们获取的图像，我们应该看到大约 90% 的精度。我们可以查看以下代码中我们预测错误类别的图像：

```py
mm = {tuple(a): b for a, b in zip(p, glob('cat/*jpg') + glob('dog/*jpg'))}
wrong = [mm[tuple(a)] for a, p, t in zip(X_test,
                                         clf.predict(X_test),
                                         y_test) if p != t]

for x in wrong:
    display(Image(x, width=150))
```

总的来说，我们的网络表现还不错。对于一些被标记为猫或狗的图像，我们也会感到困惑！

### 讨论

正如我们在 Recipe 4.3 中看到的，当我们需要在高维空间上建立分类器时，支持向量机是一个不错的选择。在这里，我们提取图像识别网络的输出，并将这些向量视为图像嵌入。我们让支持向量机找到将猫和狗分开的超平面。这对于二元情况效果很好。我们可以在有多于两个类别的情况下使用支持向量机，但情况会变得更加复杂，也许更合理的做法是在我们的网络中添加一层来完成繁重的工作。Recipe 9.7 展示了如何做到这一点。

很多时候分类器没有得到正确答案，你可以真的归咎于搜索结果的质量。在下一个配方中，我们将看看如何利用我们提取的图像特征来改进搜索结果。

## 9.6 改进搜索结果

### 问题

如何从一组图像中过滤掉异常值？

### 解决方案

将图像分类器的最高但一个层的特征视为图像嵌入，并在该空间中找到异常值。

正如我们在前一个配方中看到的，我们的网络有时无法区分猫和狗的原因之一是它看到的图像质量不太好。有时图像根本不是猫或狗的图片，网络只能猜测。

Flickr 搜索 API 不会返回与提供的文本查询匹配的图像，而是返回其标签、描述或标题与文本匹配的图像。即使是主要搜索引擎最近也开始考虑返回的图像中实际可见的内容。（因此，搜索“猫”可能会返回一张标题为“看这只大猫”的狮子图片。）

只要返回的图像大多数符合用户的意图，我们可以通过过滤掉异常值来改进搜索。对于生产系统，值得探索更复杂的东西；在我们的情况下，我们最多有几百张图像和数千个维度，我们可以使用更简单的方法。

让我们从最近的猫图片开始。由于我们按`recent`而不是`relevance`排序，我们预计搜索结果会稍微不太准确：

```py
fetch_image_set('cat', dir_name='maybe_cat', count=100, sort='recent')
```

与以前一样，我们将图像加载为一个向量：

```py
maybe_cat_fns = glob('maybe_cat/*jpg')
maybe_cats = [image.load_img(p, target_size=(224, 224))
              for p in maybe_cat_fns]
maybe_cat_vectors = np.asarray([image.img_to_array(img)
                                for img in maybe_cats])
```

我们将首先通过找到“可能是猫”的空间中的平均点来寻找异常值：

```py
centroid = maybe_cat_vectors.sum(axis=0) / len(maybe_cats)
```

然后我们计算猫向量到质心的距离：

```py
diffs = maybe_cat_vectors - centroid
distances = numpy.linalg.norm(diffs, axis=1)
```

现在我们可以看看与平均猫最不相似的东西：

