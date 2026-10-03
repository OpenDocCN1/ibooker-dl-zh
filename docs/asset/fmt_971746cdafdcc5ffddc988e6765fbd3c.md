
完成这些后，你可以从`StableDiffusionPipeline`类创建管道并将其加载到加速设备：

```py
# 2\. Load the pipeline with the chosen scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)
```

下一步是分配新的 LoRA 权重，这些是重新训练的层，决定了模型的新行为：

```py
# 3\. (Optional) Load LoRA weights
pipe.load_lora_weights("lmoroney/finetuned-misato-sd2")
```

稳定扩散支持**提示**和**负面提示**，其中第一个提示定义了图像中你想要的内容，第二个提示定义了你**不想要**的内容。以下是一个示例：

```py
# 4\. Define prompts and parameters
prompt = "(lora-misato-token) in food ad, billboard sign, 90s, anime, 
         japanese pop, japanese words, front view, plain background"

negative_prompt = (
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, 
     wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and 
     fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, 
     amputation"
)
```

负面提示非常有用，可以帮助你避免一些与 AI 生成的视觉问题，例如变形的手和脸。

接下来是定义超参数，例如推理步骤的数量、图像的大小和种子。还有一个叫做*指导尺度*的参数，它控制着你的模型有多具想象力。指导尺度值小于 5 会给模型更多的创造性自由，但模型可能不会严格遵循你的提示。指导尺度值高于 7 会使模型更紧密地遵循你的提示，但也可能导致奇怪的伪影。中间的指导尺度值 6 在自由和遵循之间提供了一个很好的平衡。没有固定的规则，所以请随意实验：

```py
num_inference_steps = 50
guidance_scale = 6.0
width = 512
height = 512
seed = 1234567
```

然后，你只需像往常一样生成图像：

```py
# 5\. Create a generator for reproducible results
generator = torch.Generator(device=device).manual_seed(seed)

# 6\. Run the pipeline
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# 7\. Save the result
image.save("lora-with-negative.png")
```

作为实验，你可以尝试使用具有相同超参数的不同调度器来产生类似的结果（见图 20-8）：

```py
# For DPMSolver, use:
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, 
            subfolder="scheduler", algorithm_type="dpmsolver++")
```

![](img/aiml_2008.png)

###### 图 20-8\. 使用不同调度器的相同提示和超参数

注意，图片中的文字完全是虚构的，但鉴于提示是关于广告的，语气相似。在左边的图片中，人物代表“孤独”和“不”，而在右边的图片中，它们暗示着“丈夫分居？”

最有趣的是角色的连贯性！例如，考虑图 20-9，其中 Misato 被画成了莫奈和毕加索的风格。我们可以看到，LoRA 学习到的特征足够一致，足以（大部分）在重新设计过程中幸存下来。

![](img/aiml_2009.png)

###### 图 20-9\. 不同风格下的角色一致性

这个例子使用了 Stable Diffusion 2，这是一个较老的模型，但使用 LoRA 很容易调整。随着你使用更先进的模型并对其进行调整，你可以得到更好的结果，但调整的时间和成本会更高。我建议从这样一个简单的模型开始，并专注于你的技艺。从那里，你可以逐步过渡到更先进的模型。

此外，Misato 的合成特性在 LoRA 重新训练中触发了不同的特征，导致从她低分辨率、高度合成的外观中创建出了新的图像。虽然这些图像在人类眼中几乎接近照片真实，但它们显然没有达到模型的要求，该模型学习到的 LoRA 本质上是 CGI 风格，并且分辨率低于训练集中的图像！

# 摘要

在本章中，您了解了如何通过使用 LoRA 和 diffusers 库来微调像 stable diffusion 这样的文本到图像模型。这项技术允许您通过一个小型自定义文件来定制特定主题或风格的模型。在这种情况下，您看到了如何调整 Stable Diffusion 2 以适应合成角色。在本章中，您还经历了所有步骤——从克隆 diffusers 到为它们创建一个包含完整自定义数据集的训练环境。您学习了如何使用训练脚本创建基于合成角色的新 LoRA，以及如何将其发布到 Hugging Face。最后，您看到了如何在推理时将 LoRA 应用于模型，以使用 Misato 角色的 LoRA 创建新颖的图像！
