
作为我们的最后一步，我们希望展示在`θ*`时，`p_θ*(y = dataset | x) = p(y = dataset | x)`，正如承诺的那样。我们已经知道`p_θ*(x, y = dataset) = p(x, y = dataset)`来自我们之前的工作。在两边都除以`p(x)`后，我们得到了期望的结果。

到目前为止，我们已经假设了一个固定的`G`，并展示了关于最优`D`的各种属性。不幸的是，在实践中我们不能假设一个固定的`G`，因为我们必须训练生成器和鉴别器。但是现在我们已经展示了一些关于最优`D`的属性，我们可以开始讨论`G`必须满足的属性，以实现全局最优——一个甚至可以完全混淆最优鉴别器的生成器。如果我们假设一个最优鉴别器，并将其得分`p_data(x) / (p_data(x) + p_g(x))`代入目标`V(G, D)`，我们得到一个仅依赖于`G`的参数或权重的目标：

```
C(G) = 𝔼_{x∼p_data(x)} [log(p_data(x) / (p_data(x) + p_g(x)))] + 𝔼_{x∼p_g(x)} [log(1 - p_data(x) / (p_data(x) + p_g(x)))]
```

```
= 𝔼_{x∼p_data(x)} [log(p_data(x) / (p_data(x) + p_g(x)))] + 𝔼_{x∼p_g(x)} [log(p_g(x) / (p_data(x) + p_g(x)))]
```

```
= 𝔼_{x∼p(x|y=dataset)} [log(p(x|y=dataset) / (p(x|y=dataset) + p_φ(x|y=generator)))] + 𝔼_{x∼p_φ(x|y=generator)} [log(p_φ(x|y=generator) / (p(x|y=dataset) + p_φ(x|y=generator)))]
```
