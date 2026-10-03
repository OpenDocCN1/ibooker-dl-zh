
# 使用 `jdeps` 分析模块依赖

执行以下命令，分析 `monitor` 模块的依赖关系（省略了部分输出）：

```
$ jdeps -summary -recursive --module-path mods:libs -m monitor
# 省略了一些模块依赖
monitor -> java.base
monitor -> monitor.observer
monitor -> monitor.observer.alpha
monitor -> monitor.observer.beta
monitor -> monitor.persistence
monitor -> monitor.rest
monitor -> monitor.statistics
monitor.observer -> java.base
monitor.observer.alpha -> java.base
monitor.observer.alpha -> monitor.observer
monitor.observer.beta -> java.base
monitor.observer.beta -> monitor.observer
monitor.persistence -> java.base
monitor.persistence -> monitor.statistics
monitor.persistence -> hibernate.jpa
monitor.rest -> java.base
monitor.rest -> monitor.statistics
monitor.rest -> spark.core
monitor.statistics -> java.base
monitor.statistics -> monitor.observer
slf4j.api -> java.base
slf4j.api -> 未找到
spark.core -> JDK 移除了内部 API
spark.core -> java.base
spark.core -> javax.servlet.api
spark.core -> jetty.server
spark.core -> jetty.servlet
spark.core -> jetty.util
spark.core -> slf4j.api
spark.core -> websocket.api
spark.core -> websocket.server
spark.core -> websocket.servlet
# 省略了更多模块依赖
```

## 其他相关选项

此外，还有一些 Java 9 和模块特定的选项。使用`--require ${modules}`，您可以列出所有需要命名模块的模块。如何使用`--jdk-internals`来分析项目的有问题依赖关系在第 7.1.2 节中解释。第 9.3.2 节解释了如何使用`--generate-module-info`和`--generate-open-module`来创建模块描述符的第一稿。如前所述，`jdeps`还会始终报告它找到的所有拆分包——这个问题在第 7.2 节中详细讨论。

## 使用 `--check` 选项

一个有趣的选项是`--check`，它提供了对模块描述符的不同视角（见图 D.2）：

- 它首先打印实际的描述符，然后是两个假设的描述符。
- 其中第一个，被描述为建议的描述符，声明了对所有在检查模块中使用类型的模块的依赖。
- 第二个，被描述为传递减少图，与第一个类似，但去除了由于暗示的可读性而可以读取的依赖（见第 9.1 节）。这意味着它是产生可靠配置的最小依赖集。
- 最后，如果模块声明了任何合格的导出（见第 9.3 节），`--check`将输出那些在可观察模块的宇宙中未使用的导出。

`--check`创建的假设描述符也可以分别使用两个选项`--list-deps`和`--list-reduced-deps`进行查看。它们也可以与类路径一起使用，在这种情况下，它们引用未命名的模块（见第 8.2 节）。

![图片](img/00041.jpeg)

图 D.2 在左侧，您可以看到`monitor.peek`（在 11.1.1 节中介绍）及其传递依赖关系，其中一些暗示了在其他模块上的可读性。在右侧，`jdeps`建议将`monitor.observer`包含在依赖列表中（因为其类型被直接引用）。此外，它列出了`monitor.peek`需要依赖的最小模块集，以充分利用暗示的可读性。
