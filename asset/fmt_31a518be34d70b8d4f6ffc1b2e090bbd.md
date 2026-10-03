
```py
from diffusers.optimization import get_scheduler

num_epochs=100                                             ①
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,
    betas=(0.95,0.999),weight_decay=0.00001,eps=1e-8)      ②
lr_scheduler=get_scheduler(                                ③
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=300,
    num_training_steps=(len(train_dataloader) * num_epochs))
```

① 将训练模型 100 个周期

② 使用 `AdamW` 优化器

③ 使用 `diffusers` 库中的学习率调度器来控制学习率

`get_scheduler()` 函数的精确定义由 Hugging Face 在 GitHub 上定义：[`mng.bz/ZVo5`](https://mng.bz/ZVo5)。在前 300 个训练步骤（预热步骤）中，学习率从 0 线性增加到 0.0001（我们在 `AdamW` 优化器中设置的学习率）。在 300 步之后，学习率按照 0.0001 和 0 之间的余弦函数值递减。在下面的列表中，我们训练模型 100 个周期。

列表 15.6 训练去噪 U-Net 模型

```py
for epoch in range(num_epochs):
    model.train()
    tloss = 0
    print(f"start epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["input"].to(device)*2-1
        nums = clean_images.shape[0]
        noise = torch.randn(clean_images.shape).to(device)
        timesteps = torch.randint(0,
                noise_scheduler.num_train_timesteps,
                (nums, ),
                device=device).long()
        noisy_images = noise_scheduler.add_noise(clean_images,
                     noise, timesteps)                         ①

noise_pred = model(noisy_images, 
                       timesteps)["sample"]                    ②
        loss=torch.nn.functional.l1_loss(noise_pred, noise)    ③
        loss.backward()
        optimizer.step()                                       ④
        lr_scheduler.step()
        optimizer.zero_grad()
        tloss += loss.detach().item()
        if step%100==0:
            print(f"step {step}, average loss {tloss/(step+1)}")
torch.save(model.state_dict(),'files/diffusion.pth')
```

① 在训练集中向干净图像添加噪声

② 使用去噪 U-Net 预测噪声图像中的噪声

③ 将预测的噪声与实际噪声进行比较以计算损失

④ 调整模型参数以最小化平均绝对误差

在每个周期中，我们遍历训练集中所有干净的花朵图像的批次。我们向这些干净的图像添加噪声，并将它们输入到去噪 U-Net 中以预测这些图像中的噪声。然后我们将预测的噪声与实际噪声进行比较，并调整模型参数以最小化两者之间的平均绝对误差（像素级）。

这里描述的训练过程需要几个小时，使用 GPU 训练。训练后，训练好的模型权重将保存在您的计算机上。或者，您可以从我的网站 [`mng.bz/RNlD`](https://mng.bz/RNlD) 下载训练好的权重。下载后请解压文件。

### 15.4.2 使用训练好的模型生成花朵图像

为了生成花朵图像，我们将使用 50 个推理步骤。这意味着我们将查看 `t = 0` 和 `t = T` 之间等间隔的 50 个时间步，在我们的情况下 `T = 1,000`。因此，50 个推理时间步是 `t = 980, 960, 940, . . . , 20, 0`。我们将从纯随机噪声开始，这对应于 `t = 1000` 的图像。我们使用训练好的去噪 U-Net 模型对其进行去噪，并在 `t = 980` 创建一个噪声图像。然后我们将 `t = 980` 的噪声图像呈现给训练好的模型进行去噪，以获得 `t = 960` 的噪声图像。我们重复这个过程多次，直到获得 `t = 0` 的图像，这是一个干净的图像。这个过程通过在本地模块 `ch15util.py` 中的 `DDIMScheduler()` 类的 `generate()` 方法实现。

列表 15.7 在 `DDIMScheduler()` 类中定义 `generate()` 方法

```py
    @torch.no_grad()
    def generate(self,model,device,batch_size=1,generator=None,
         eta=1.0,use_clipped_model_output=True,num_inference_steps=50):
        imgs=[]
        image=torch.randn((batch_size,model.in_channels,model.sample_size,

model.sample_size),
            generator=generator).to(device)              ①

        self.set_timesteps(num_inference_steps)
        for t in tqdm(self.timesteps):                   ②
            model_output = model(image, t)["sample"]     ③
            image = self.step(model_output,t,image,eta,
                  use_clipped_model_output=\
                  use_clipped_model_output)              ④
            img = unnormalize_to_zero_to_one(image)
            img = img.cpu().permute(0, 2, 3, 1).numpy()
            imgs.append(img)                             ⑤
        image = unnormalize_to_zero_to_one(image)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return {"sample": image}, imgs 
```

① 使用随机噪声作为起点（即 `t = 1,000` 的图像）

② 使用 50 个推理时间步（`t = 980, 960, 940, . . ., 20, 0`）

③ 使用训练好的去噪 U-Net 模型来预测噪声

④ 根据预测的噪声创建图像

⑤ 将中间图像保存在列表 `imgs` 中

在这个 `generate()` 方法中，我们同样创建了一个列表 `imgs`，用于存储所有在时间步 `t = 980, 960, . . . , 20, 0` 时的中间图像。我们将使用它们来可视化去噪过程。`generate()` 方法返回一个包含生成的图像和列表 `imgs` 的字典。

接下来，我们将使用之前的 `generate()` 方法创建 10 张干净的图像。

列表 15.8 使用训练好的去噪 U-Net 模型生成图像

```py
sd=torch.load('files/diffusion.pth',map_location=device)
model.load_state_dict(sd)
with torch.no_grad():
    generator = torch.manual_seed(1)                     ①
    generated_images,imgs = noise_scheduler.generate(
        model,device,
        num_inference_steps=50,
        generator=generator,
        eta=1.0,
        use_clipped_model_output=True,
        batch_size=10)                                   ②
imgnp=generated_images["sample"]    
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4),dpi=300)
for i in range(10):                                      ③
    ax = plt.subplot(2,5, i + 1)
    plt.imshow(imgnp[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
plt.show()  
```

① 设置随机种子为 1，以便结果可重复

② 使用定义的 `generate()` 方法创建 10 张干净的图像

③ 绘制生成的图像

我们将随机种子设置为 1。因此，如果您使用我网站上的训练模型，您将得到与图 15.6 中显示的相同的结果。我们使用之前定义的 `generate()` 方法，使用 50 次推理步骤创建 10 张干净的图像。然后我们将这 10 张图像以 2×5 的网格形式绘制出来，如图 15.6 所示。

![图片](img/CH15_F06_Liu.png)

图 15.6 由训练好的去噪 U-Net 模型创建的花朵图像。

如您从图 15.6 中看到的那样，生成的花朵图像看起来很真实，类似于训练数据集中的图像。

练习 15.1

修改代码列表 15.8 并将随机种子改为 2。保持其余代码不变。重新运行代码列表，看看生成的图像是什么样子。

`generate()` 方法还返回一个包含 50 个中间步骤中所有图像的列表 `imgs`。我们将使用它们来可视化去噪过程。

列表 15.9 可视化去噪过程

```py
steps=imgs[9::10]                                ①
imgs20=[]
for j in [1,3,6,9]:
    for i in range(5):
        imgs20.append(steps[i][j])               ②
plt.figure(figsize=(10,8),dpi=300)
for i in range(20):                              ③
    k=i%5
    ax = plt.subplot(4,5, i + 1)
    plt.imshow(imgs20[i])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.title(f't={800-200*k}',fontsize=15,c="r")
plt.show()
```

① 保持时间步 800、600、400、200 和 0

② 从 10 组花朵中选择 4 组

③ 以 4×5 的网格绘制 20 张图像

列表中的 `imgs` 包含所有 50 个推理步骤中的 10 组图像，`t = 980, 960, ...，20, 0`。因此，列表中共有 500 张图像。我们选择了五个时间步（`t = 800, 600, 400, 200 和 0`）用于四种不同的花朵（图 15.6 中的第 2、4、7 和 10 张图像）。然后我们将这 20 张图像以 4×5 的网格形式绘制出来，如图 15.7 所示。

![图片](img/CH15_F07_Liu.png)

图 15.7 训练好的去噪 U-Net 模型如何逐步将随机噪声转换为干净的鲜花图像。我们将随机噪声输入到训练好的模型中，以获得时间步 980 的图像。然后我们将 `t = 980` 的噪声图像输入到模型中，以获得 `t = 960` 的图像。我们重复这个过程 50 次推理步骤，直到我们获得 `t = 0` 的图像。该图的第一列显示了 `t = 800` 时的四个花朵；第二列显示了相同的四个花朵在 `t = 600` 时...；最后一列显示了 `t = 0`（即干净的鲜花图像）的四个花朵。

图 15.7 的第一列显示了 `t = 800` 时的四个花朵图像。它们接近随机噪声。第二列显示了 `t = 600` 时的花朵，它们开始看起来像花朵。随着我们向右移动，图像变得越来越清晰。最右边的一列显示了 `t = 0` 时的四个干净花朵图像。

现在您已经了解了扩散模型的工作原理，我们将讨论文本到图像的生成。文本到图像的 Transformers，如 DALL-E 2、Imagen 和 Stable Diffusion 的图像生成过程与我们在本章前面讨论的逆向扩散过程非常相似，只是在生成图像时，模型将文本嵌入作为条件信号。

## 15.5 文本到图像的 Transformers

如 OpenAI 的 DALL-E 2、Google 的 Imagen 和 Stability AI 的 Stable Diffusion 这样的文本到图像 Transformer 使用扩散模型从文本描述生成图像。这些文本到图像 Transformer 的一个重要组成部分是扩散模型。文本到图像生成的过程涉及将文本输入编码成一个潜在表示，然后将其用作扩散模型的条件信号。这些 Transformer 通过迭代地去除随机噪声向量来学习生成与文本描述相对应的逼真图像，这个过程由编码的文本引导。

所有这些文本到图像的 Transformer 的关键在于一个能够理解不同模态内容的模型。在这种情况下，该模型必须理解文本描述并将它们与图像以及反之联系起来。

在本节中，我们将以 OpenAI 的 CLIP 模型为例。CLIP 是 DALL-E 2 的关键组件。我们将讨论 CLIP 是如何被训练来理解文本描述和图像之间的联系的。然后，我们使用一个简短的 Python 程序，通过使用 OpenAI 的 DALL-E 2，从文本提示中生成图像。

### 15.5.1 CLIP：一个多模态 Transformer

近年来，计算机视觉和自然语言处理（NLP）的交叉领域取得了显著进展，其中之一是 OpenAI 创建的 CLIP 模型。这个创新模型旨在理解并解释自然语言环境中的图像，这一能力在图像生成和图像分类等众多应用中具有巨大的潜力。

CLIP 模型是一个多模态 Transformer，它弥合了视觉数据和文本数据之间的差距。它通过将图像与相应的文本描述关联起来来训练理解图像。与需要显式标记图像的传统模型不同，CLIP 使用大量图像及其自然语言描述的数据集来学习视觉概念的更一般化表示。

![图片](img/CH15_F08_Liu.png)

图 15.8 OpenAI 的 CLIP 模型是如何训练的。收集了一个大规模的文本-图像对训练数据集。模型的文本编码器将文本描述压缩成 D 值文本嵌入。图像编码器将相应的图像转换成具有 D 值的图像嵌入。在训练过程中，一批 N 个文本-图像对被转换成 N 个文本嵌入和 N 个图像嵌入。CLIP 使用对比学习方法来最大化配对嵌入之间的相似性（图中对角线值的总和）同时最小化来自不匹配文本-图像对的嵌入之间的相似性（图中非对角线值的总和）。

CLIP 模型的训练，如图 15.8 所示，始于收集包含图像及其相关文本描述的大规模数据集。OpenAI 利用了多样化的来源，包括公开可用的数据集和网页爬取的数据，以确保有广泛的视觉和文本内容。然后对数据集进行预处理，以标准化图像，使它们都具有相同的形状，并对文本进行分词，为输入模型做准备。

CLIP 采用双编码器架构，包括图像编码器和文本编码器。图像编码器处理输入图像，而文本编码器处理相应的文本描述。这些编码器将图像和文本投影到一个共享的嵌入空间中，在那里它们可以被比较和对齐。

CLIP 训练的核心在于其对比学习方法。对于数据集中每个包含 N 个图像-文本对的批次，模型旨在最大化配对嵌入之间的相似性（如图 15.8 中对角线值的总和），同时最小化来自不匹配的文本-图像对嵌入之间的相似性（非对角线值的总和）。图 15.9 展示了如何基于文本提示生成逼真图像的文本到图像 Transformer，如 DALL-E 2。

![图片](img/CH15_F09_Liu.png)
