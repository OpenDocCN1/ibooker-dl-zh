
# 性能

使用多模块编译，`ServiceMonitor` 应用程序的构建时间不到一秒，这比逐步构建七个模块快四倍。但这是一个相当极端的例子，因为每个模块只包含两到三个类。相对而言，启动编译器七次的开销很大；但从绝对意义上来说，这仅仅减少了三秒钟。考虑到任何适度规模项目的构建时间，减少几秒钟几乎不值得使构建更加复杂。

# 弱循环依赖

在这种情况下，如果构建过程中应该没有警告，那么就没有办法绕过多模块编译。

> 多模块编译是可选的，其好处并不足以推荐它作为默认实践。尤其是如果你的工具不支持无缝集成，设置它可能不值得付出努力。这是一个典型的“视情况而定”的情况。不过，我必须说，我喜欢它在更高层次上操作：模块而不是仅仅类型。

## 4.4 编译器选项

模块系统带来了许多新的命令行选项，本书中对此进行了详细解释。为了确保你可以轻松找到它们，表 4.1 列出了所有与编译器相关的选项。查看[`docs.oracle.com/javase/9/tools/javac.htm`](https://docs.oracle.com/javase/9/tools/javac.htm)以获取官方编译器文档。

表 4.1 所有模块相关编译器（`javac`命令）选项的字母顺序表。描述基于文档，引用指向本书中解释如何使用选项的章节。

| 选项 | 描述 | 参考 |
| --- | --- | --- |
| `--add-exports` | 允许模块导出额外的包 | 11.3.4 |
| `--add-modules` | 除了初始模块外，还定义根模块 | 3.4.3 |
| `--add-reads` | 在模块之间添加读取边 | 3.4.4 |
| `--limit-modules` | 限制可观察模块的宇宙 | 5.3.5 |
| `--module`, `-m` | 设置初始模块 | 4.3.5 |
| `--module-path`, `-p` | 指定查找应用程序模块的位置 | 3.4 |
| `--module-source-path` | 传达项目的目录结构 | 4.3.2 |
| `--module-version` | 指定编译中模块的版本 | 13.2.1 |
| `--patch-module` | 在编译过程中用类扩展现有模块 | 7.2.4 |
| `--processor-module-path` | 指定查找注解处理器模块的位置 | 4.2.1 |
| `--system` | 覆盖系统模块的位置 | |
| `--upgrade-module-path` | 定义可升级模块的位置 | 6.1.3 |

### 新的`--release`选项

你是否曾经使用过`-source`和`-target`选项来编译你的代码，以便在较旧的 Java 版本上运行，但看到它在运行时崩溃，因为一个方法调用失败，出现了一个看似无法解释的错误？也许你忘记指定了`-bootclasspath`。

没有这个选项，编译器会创建 JVM 可以理解的字节码（好），但它会链接到当前版本的核心库 API（不好）。这可能导致调用在旧 JDK 版本中不存在的类型或方法，从而引起运行时错误。

从 Java 9 开始，编译器通过`--release`选项防止了这种常见的操作错误，该选项将所有三个选项设置为正确的值。

## 4.5 打包模块化 JAR

在从想法到运行代码的过程中，编码和编译之后的下一步是将类文件打包成一个模块。正如 3.1.2 节所解释的，这应该生成一个模块化的 JAR 文件，它就像一个普通的 JAR 文件一样，但包含模块的描述符`module-info.class`。因此，你期望可信的`jar`工具负责打包。创建一个模块化的 JAR 文件（在这种情况下为`monitor.observer`）就是这么简单：

```
$ jar --create --file mods/monitor.observer.jar -C monitor.observer/target/classes .
```

将新的命令行别名放在一边，这个调用与 Java 9 之前完全相同。有趣且隐含的细节是，因为 `monitor.observer/target/classes` 包含一个 `module-info.class`，所以生成的 `monitor.observer.jar` 也将包含它，使其成为一个模块化 JAR。

虽然 `jar` 工具的工作方式与之前相似，但有一些与模块相关的细节和新增功能，例如定义模块的入口点，我们应该看看。

> 注意：JAR 不是唯一用于交付 Java 字节码的格式。JEE 还与 WAR 和 EAR 文件一起工作。尽管如此，在规范更新以接受模块之前，无法创建模块化的 WAR 或 EAR。

### 4.5.1 JAR 的快速回顾

为了确保我们都在同一页面上，让我们快速看一下 `jar` 是如何用于打包存档的。正如我刚才指出的，如果包含的文件列表中包含模块描述符 `module-info.class`，则结果是一个模块化 JAR。

让我们以打包 `monitor.observer` 的命令为例。结果是 `mods` 中的 `module.observer.jar`，它包含 `monitor.observer/target/classes` 及其子目录中的所有类文件。因为 `classes` 包含一个模块描述符，所以 JAR 也将包含它，因此无需额外努力就是一个模块化 JAR：

```
$ jar --create ① --file mods/monitor.observer.jar ② -C monitor.observer/target/classes . ③
```

> ① 此操作模式表示创建一个存档（另一种选择是 `-c`）。
> ② 要创建的存档文件名（另一种选择是 `-f`）
> ③ `-C` 使 `jar` 变为指定的文件夹，点号 (`.`) 告诉它包含文件夹中的所有源文件。

在打包模块时，您应该考虑使用 `--module-version` 记录模块的版本。第 13.2.1 节解释了如何进行操作。

### 4.5.2 分析 JAR

当与 JAR 一起工作时，了解分析您所创建内容的方法很有帮助。特别是，了解 JAR 包含的文件以及其模块描述符的内容非常重要。幸运的是，`jar` 有选项可以做到这两点。

#### 列出 JAR 的内容

最明显的事情是查看 JAR 的内容，这可以通过 `--list` 实现。以下片段显示了上一节创建的 `monitor.observer.jar` 的内容。它包含一个 `META-INF` 文件夹，我们不会深入探讨，因为它已经存在多年，并且与模块系统无关。还有一个模块描述符，以及 `monitor.observer` 包中的 `DiagnosticDataPoint` 和 `ServiceObserver` 类。没有什么特别或意外的：

```
$ jar --list --file mods/monitor.observer.jar
> META-INF/
> META-INF/MANIFEST.MF
> module-info.class
> monitor/
> monitor/observer/
> monitor/observer/DiagnosticDataPoint.class
> monitor/observer/ServiceObserver.class
```

这不是一个新命令——它只是因为新的别名而看起来不同：`--list`是`-t`的简写，`--file`是`-f`的简写。在 Java 9 之前，`jar -t -f some.jar`会做同样的事情。

#### 检查模块描述符

模块描述符是一个类文件，因此由字节码组成。这使得使用工具查看其内容成为必要。幸运的是，`jar`可以使用`--describe-module`（或`-d`）来做这件事。检查`monitor.observer.jar`，你会发现它是一个名为`monitor.observer`的模块，导出同名的包并需要基础模块：

```
$ jar --describe-module --file mods/monitor.observer.jar
> monitor.observer jar:.../monitor.observer.jar/!module-info.class
> exports monitor.observer
> requires java.base mandated
```

（如果你想知道`mandated`是什么意思，记得从 3.1.4 节中了解到，每个模块隐式地需要基础模块，这意味着必须存在`java.base`。）

### 4.5.3 定义入口点

要启动一个 Java 应用程序，必须知道入口点，它是包含`public static void main(String[])`方法的类之一。包含该方法的类可以在应用程序启动时在命令行上指定，或者记录在随 JAR 文件一起提供的清单文件中。如果你不知道这些选项是如何工作的，不必担心，因为 Java 9 增加了一个第三种方式，即使用模块的方式。

当使用`jar`将类文件打包到归档中时，你可以使用`--main-class ${class}`定义主类，其中`${class}`是包含`main`方法的类的完全限定名（意味着包名后跟一个点和类名）。它将被记录在模块描述符中，并在模块是启动应用程序的初始模块时默认用作主类（有关详细信息，请参阅 5.1 节）。

> 注意：如果你习惯于设置清单的`Main-Class`条目来创建可执行 JAR 文件，你将很高兴地听到`jar --main-class`也会设置它。

`ServiceMonitor` 应用程序在`monitor.Main`中有一个单一的入口点。你可以使用`--main-class monitor.Main`来记录在打包过程中的情况：

```
$ jar --create --file mods/monitor.jar --main-class monitor.Main -C monitor/target/classes .
```

使用`--describe-module`，你可以看到主类已被记录在描述符中：

```
$ jar --describe-module --file mods/monitor.jar
> monitor jar:.../monitor.jar/!module-info.class # requires and contains truncated
> main-class monitor.Main
```

很有趣的是，`jar`工具既没有验证你声称存在此类的能力，也没有责任。没有检查它是否存在或是否包含合适的`main`方法。如果出现问题，现在不会发生错误，但启动模块将失败。

### 4.5.4 归档器选项

我们只探索了 `jar` 提供的最重要选项。其他一些选项在不同的上下文中变得有趣，并在相关章节中解释。为了确保您可以轻松找到它们，表 4.2 列出了与模块系统相关的选项。访问 [`docs.oracle.com/javase/9/tools/jar.htm`](https://docs.oracle.com/javase/9/tools/jar.htm) 获取官方 `jar` 文档。

表 4.2 所有模块相关归档器（`jar` 命令）选项的字母顺序表。描述基于文档，参考信息指向本书中解释如何使用选项的章节。

| 选项 | 描述 | 参考信息 |
| --- | --- | --- |
| `--hash-modules` | 记录依赖模块的哈希值 | |
| `--describe-module`, `-d` | 显示模块的名称、依赖项、导出、包等 | 4.5.2 |
| `--main-class` | 应用程序入口点 | 4.5.3 |
| `--module-path`, `-p` | 指定查找应用程序模块以记录哈希值的位置 | 3.4 |
| `--module-version` | 指定编译中模块的版本 | 13.2.1 |
| `--release` | 创建包含不同 Java 版本字节码的多版本 JAR 文件 | 附录 E |
| `--update` | 更新现有归档，例如通过添加更多类文件 | 9.3.3 |

## 摘要

+ 确保选择满足项目要求的目录结构。如果有疑问，请坚持使用构建系统的默认结构。
+ 编译一个模块的所有源代码（包括声明）的 `javac` 命令与 Java 9 之前相同，除了它使用模块路径而不是类路径。
+ 模块源路径（`--module-source-path`）通知编译器项目是如何结构的。这使编译器操作从处理类型提升到处理模块，允许您使用简单选项（`--module` 或 `-m`）编译所选模块及其所有依赖项，而不是列出源文件。
+ 模块化 JAR 文件只是带有模块描述符 `module-info.class` 的 JAR 文件。`jar` 工具对它们处理得和其他类文件一样好，因此将它们全部打包到 JAR 文件中不需要任何新选项。
+ 可选地，`jar` 允许指定模块的入口点（使用 `--main-class`），这是具有 `main` 方法的类。这使得启动模块变得简单。

# 5 运行和调试模块化应用程序

## 本章涵盖

+ 通过指定初始模块启动模块化应用程序
+ 从模块中加载资源
+ 验证模块、模块集和模块图
+ 减少和列出可观察模块的宇宙
+ 使用日志调试模块化应用程序

如第三章和第四章所述，将模块定义、编译并打包成模块化 JAR 文件后，终于到了使用`java`命令启动 JVM 和运行应用程序的时候了。这为我们提供了讨论与运行时相关的概念的机会：如何从模块中加载资源（第 5.2 节）。然而，迟早会遇到问题，因此第 5.3 节还将探讨使用各种命令行选项调试模块配置。

到本章结束时，你将能够启动由模块组成的程序。除此之外，你将深刻理解模块系统如何处理给定的配置，以及如何通过日志和其他诊断工具来观察这一点。

这也完成了本书的第一部分，它教授了你编写、编译和运行简单模块化应用程序所需的一切知识。它为第二部分和第三部分将要探讨的更高级功能奠定了基础，其中最重要的是那些支持逐步迁移到模块系统的功能。

## 5.1 使用模块启动 JVM

在定义模块依赖关系和 API、创建模块化 JAR 文件并将它们放置在模块路径上之后，使用模块化应用程序启动 JVM 竟然如此简单。你只需要指定初始模块，也许还需要指定主类。

`java`命令有一个`--module ${module}`选项，它指定了初始模块`${module}`。模块解析从这里开始，并且这也是将启动具有`public static void main`方法的 main 类的模块。

特定的类要么由初始模块的描述符定义，要么通过在模块名称后附加一个斜杠和完全限定的类名（见 5.1.1 节）来使用`--module ${module}/${class}`指定。

对于 `ServiceMonitor` 应用程序，所有准备工作最终都集中在你已经看到的调用上，该调用以 `monitor` 作为初始模块启动 JVM：

```
$ java --module-path mods:libs --module monitor
```

如第 3.4 节所述，`--module-path mods:libs`通知模块系统`mods`和`libs`目录包含 `ServiceMonitor` 的应用程序模块。选项`--module monitor`将 `monitor` 定义为初始模块，因此模块系统将解析 `monitor` 的所有依赖关系，并构建如前所述的模块图。然后，它将启动在第 4.5.3 节打包期间在模块描述符中设置的 main 类：`monitor.Main`。

### 5.1.1 指定主类

`--module`选项也可以用来定义应用程序的主类。为此，初始模块的名称后面跟着一个正斜杠和完全限定的类名（包名后跟一个点和类名）。

在这里，你明确地定义了应用程序是通过在监控器的类`monitor.Main`中调用`main`方法来启动的：

```
$ java --module-path mods:libs --module monitor/monitor.Main
```

在命令行上指定主类覆盖了模块描述符中定义的内容。这意味着应用程序仍然可以有多个入口点，就像没有模块系统一样。如果其中一个是合理的默认值，那么将其嵌入到模块描述符中是有意义的，如第 4.5.3 节所述。

如果监控器将 `monitor.Main` 定义为其主类，但出于某种原因你不想使用它，你可以轻松地覆盖它。使用以下命令，应用程序通过调用监控器的 `some.other.MainClass` 来启动，忽略监控器描述符中定义的任何内容：

```
$ java --module-path mods:libs --module monitor/some.other.MainClass
```

为了使这起作用，初始模块必须包含指定的类。由于 `monitor` 和 `some.other.MainClass` 的情况并非如此，执行你刚才看到的命令会导致错误：

```
> 错误：无法找到或加载主类
> some.other.MainClass 在模块 monitor 中
```

### 5.1.2 如果初始模块和主模块不是同一个
