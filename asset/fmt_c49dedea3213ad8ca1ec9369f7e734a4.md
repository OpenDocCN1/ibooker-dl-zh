

# 第二部分 适应现实世界的项目

本书第一部分探讨了模块系统的基本知识以及如何编译、打包和运行模块化应用程序。除了教授相关机制外，它还展示了未来的 Java 项目将如何组织和发展。

但对于现有的项目呢？我相信你们希望看到它们在 Java 9 或更高版本上运行，也许甚至作为模块。这部分将介绍如何实现这一点。

第一步，让项目在 Java 9+ 上编译和运行，对于任何不想在 Java 8 的生命周期结束后继续使用或支付支持费用的代码库来说是强制性的。第二步，将项目的工件转换为模块化 JAR 文件，是可选的，并且可以逐步完成。

第六章和第七章专门介绍迁移到 Java 9。它们都是关于让你的非模块化、基于类路径的项目在新版本上工作（而不创建任何模块）。第八章介绍了允许你逐步模块化项目的功能。第九章提供了一些战略建议，教你如何使用第六章至第八章中学到的知识迁移和模块化你的项目。

我建议按照这个顺序阅读这些章节，但如果你更喜欢只在需要时研究技术细节，你可以从第九章开始。或者，你也可以先了解你可能会遇到的最大挑战：对 JEE 模块的依赖（第 6.1 节）和对 JDK 内部的依赖（第 7.1 节）。

本书本部分没有针对所有内容提供具体的示例。在[`github.com/CodeFX-org/demo-java-9-migration`](https://github.com/CodeFX-org/demo-java-9-migration)的存储库中包含了一个 `ServiceMonitor` 应用程序的变体，其中包含一些需要修复才能在 Java 9+ 上运行的问题。试试看吧！

# 6 移动到 Java 9 或更高版本时的兼容性挑战

本章涵盖

- 为什么 JEE 模块被弃用且默认不解析
- 编译和运行依赖于 JEE 模块的代码
- 为什么 `URLClassLoader` 的转换失败
- 理解新的 JDK 运行时图像布局
- 替换已删除的扩展机制、支持标准覆盖机制和引导类路径选项

本章和第七章讨论了将现有代码库迁移到 Java 9 及更高版本时的兼容性挑战。你目前不会创建任何模块；这些章节是关于在最新版本上构建和运行现有项目。

为什么移动到 Java 9+ 需要两整个章节？难道你不能安装最新的 JDK 并期望一切都能正常工作吗？Java 不是应该向后兼容的吗？是的——如果你的项目及其依赖项只依赖于非弃用、标准化、有文档记录的行为。但这只是一个大前提，而且结果证明，在没有任何强制措施的情况下，更广泛的 Java 社区已经偏离了这条道路。

正如你将在本章中看到的，模块系统弃用了一些 Java 特性，移除了其他特性，并更改了一些内部结构：

- 包含 JEE API 的模块已弃用，需要手动解决（第 6.1 节）。
- 应用程序类加载器（也称为系统类加载器）不再是 `URLClassLoader`，这破坏了一些类型转换（第 6.2 节）。
- Java 运行时图像（JRE 和 JDK）的目录布局已彻底翻新（第 6.3 节）
- 一些机制，如紧凑配置文件和 endorsed-standards 覆盖机制已被移除（第 6.4 节）。
- 还有一些小的变化，比如不再允许单下划线作为标识符（第 6.5 节）。

但这还不是全部。第七章讨论了两个更多挑战（内部 API 和分割包）。它们各自有自己的章节，因为在你迁移项目后，你可能会再次遇到它们与非 JDK 模块一起。

这些变化一起破坏了一些库、框架、工具、技术和可能也是你的代码，因此不幸的是，更新到 Java 9+ 并不总是容易的任务。一般来说，项目越大、越老，它需要的工作就越多。然而，这通常是值得的投资，因为它是一个偿还技术债务并使代码库变得更好的机会。

到本章和下一章结束时，你将了解更新到 Java 9、10、11 甚至更晚版本的挑战。给定一个应用程序，你将能够对需要做什么做出明智的猜测；并且假设所有依赖项都配合，你将能够在最新版本上使其工作。你还将为第九章做好准备，该章节讨论了迁移到 Java 9 及以后的策略。

> **关于类路径**
>
> 第八章详细讨论了非模块化代码如何在模块化的 JDK 上运行。目前你只需要知道以下内容：
>
> - 类路径仍然完全有效。在迁移到 Java 9+ 的过程中，你将继续使用它而不是模块路径。
> - 即使如此，模块系统仍然在发挥作用：例如，关于模块解析。
> - 类路径上的代码将自动读取大多数模块（但并非全部：请参阅第 6.1 节），因此它们在编译时或运行时无需额外配置即可使用。

## 6.1 与 JEE 模块一起工作

Java SE 中的许多代码都与 Java EE / Jakarta EE（我将其缩写为 JEE）相关：CORBA 就是其中之一，Java Architecture for XML Binding (JAXB)和 Java API for XML Web Services (JAX-WS)也是如此。这些和其他 API 最终出现在表 6.1 中显示的六个模块中。这可能会是一个微不足道的旁白和故事的结尾，但不幸的是并非如此。当你尝试编译或运行依赖于这些模块中的类的代码时，模块系统会声称这些模块在图中缺失。

在 Java 9 中，对于使用 `java.xml.bind` 模块中的 `JAXBException` 的类，存在一个编译错误：

```
> 错误：包 javax.xml.bind 不可见
> import javax.xml.bind.JAXBException;
> ^
> （包 javax.xml.bind 在模块 java.xml.bind 中声明，
> 该模块不在模块图中）
> 1 个错误
```

如果你通过了编译器，但忘记了调整运行时，你会得到一个 `NoClassDefFoundError` 错误：

```
> 线程"main"中的异常 java.lang.NoClassDefFoundError:
> javax/xml/bind/JAXBException
> at monitor.Main.main(Main.java:27)
> 原因：ClassNotFoundException:
> javax.xml.bind.JAXBException
> at java.base/BuiltinClassLoader.loadClass
> (BuiltinClassLoader.java:582)
> at java.base/ClassLoaders$AppClassLoader.loadClass
> (ClassLoaders.java:185)
> at java.base/ClassLoader.loadClass
> (ClassLoader.java:496)
> ... 更多
```

发生了什么？为什么在类路径上的代码中没有正确标准化的 Java API，对此能做些什么？

**表 6.1 六个 JEE 模块。描述引用了文档。**

| 模块名称 | 描述 | 包 |
| --- | --- | --- |
| `java.activation` | 定义了 JavaBeans 激活框架（JAF）API | `javax.activation` |
| `java.corba` | 定义了 Open Management Group（OMG）CORBA API 的 Java 绑定和 RMI-IIOP API | `javax.activity`, `javax.rmi`, `javax.rmi.CORBA`, `org.omg.*` |
| `java.transaction` | 定义了 Java 事务 API（JTA）的一个子集，以支持 CORBA 互操作 | `javax.transaction` |
| `java.xml.bind` | 定义了 JAXB API | `javax.xml.bind.*` |
| `java.xml.ws` | 定义了 JAX-WS 和 Web 服务元数据 API | `javax.jws`, `javax.jws.soap`, `javax.xml.soap`, `javax.xml.ws.*` |
| `java.xml.ws.annotation` | 定义了 Common Annotations API 的一个子集，以支持在 Java SE 平台上运行的程序 | `javax.annotation` |

### 6.1.1 为什么 JEE 模块是特殊的？

Java SE 包含一些由认可的规范和独立技术组成的包。这些技术是在 Java 社区进程（JCP）之外开发的，通常因为它们依赖于由其他机构管理的标准。例如，由万维网联盟（W3C）和 Web 超文本应用技术工作组（WHATWG）开发的文档对象模型（DOM），以及简单 XML API（SAX）。如果你感兴趣，你可以在[`mng.bz/8Ek7`](http://mng.bz/8Ek7)找到它们的列表和它们所在的包。其中许多不均衡地落入表 6.1 中列出的 JEE 模块：`java.corba`、`java.xml.bind` 和 `java.xml.ws`。

从历史上看，Java 运行时环境（JRE）附带这些技术的实现，但准备让用户独立于 JRE 进行升级。这可以通过认可的规范覆盖机制（见第 6.5.3 节）来完成。

类似地，应用服务器通常通过提供自己的实现来扩展或升级 CORBA、JAXB 或 JAX-WS API，以及 JavaBeans 激活框架（在 `java.activation` 中）或 JTA（在 `java.transaction` 中）。最后，`java.xml.ws.annotation` 包含了 `javax.annotation` 包。它通常被各种 JSR 305 实现扩展，这些实现最著名的是它们与 `null` 相关的注解。

在所有这些扩展或替换 Java 附带 API 的案例中，技巧是使用完全相同的包和类名，这样类就会从外部 JAR 加载而不是内置的。在模块系统的术语中，这被称为拆分包：相同的包在不同的模块或模块和类路径之间拆分。

> **拆分包的终结**
>
> 在 Java 9+ 及以后的版本中，拆分包不再有效。我们将在第 7.2 节中详细探讨这个问题——现在只需知道，与 Java 一起分发的包中的类路径上的类实际上是不可见的：
>
> - 如果 Java 包含一个具有相同完全限定名的类，那么将加载那个类。
> - 如果 Java 内置的包版本不包含所需的类，结果将是前面展示的编译错误或 `NoClassDefFoundError`。而且这无论类是否存在于类路径上都会发生。

这是一个适用于所有模块的所有包的通用机制：将它们在模块和类路径之间拆分使得类路径部分不可见。使六个 JEE 模块特殊的是，与其他模块不同，通常使用拆分包方法来扩展或升级它们。

为了使应用服务器和像 JSR 305 实现这样的库在没有大量配置的情况下工作，做出了一些权衡：对于类路径上的代码，Java 9 和 10 默认不解析 JEE 模块，这意味着它们不会进入模块图，因此不可用（有关未解析的模块，请参阅第 3.4.3 节；有关类路径场景的详细信息，请参阅第 8.2.2 节）。

这对于带有这些 JEE API 自己实现的程序来说效果很好，但对于依赖于 JDK 变体的程序来说就不那么好了。没有进一步的配置，使用这些六个模块中的类型在类路径上的代码将无法编译和运行。

为了消除这种复杂性并正确地将 Java SE 与 JEE 分开，这些模块在 Java 9 中被弃用，在 Java 11 中被移除。随着它们的移除，命令行工具如 `wsgen` 和 `xjc` 也不再随 JDK 一起提供。

### 6.1.2 手动解析 JEE 模块

如果由于缺少 JEE API 而导致编译或运行时错误，或者如果 JDeps 分析（见附录 D）显示你依赖于 JEE 模块，你会怎么做？有三个答案：

- 如果你的应用程序在应用程序服务器中运行，它可能提供了这些 API 的实现，在这种情况下，你可能不会遇到运行时错误。根据你的配置，你可能需要修复构建错误，不过——其他两种解决方案都应该能解决这个问题。
- 选择该 API 的第三方实现，并将其作为依赖项添加到你的项目中。由于 JEE 模块默认不解析，该实现将在编译和运行时无问题地使用。
- 在 Java 9 和 10 中，如第 3.4.3 节所述，使用 `--add-modules` 添加平台模块。由于 Java 11 中移除了 JEE 模块，这在那里将不起作用。

该节开头提供的示例尝试使用 `java.xml.bind` 模块中的 `JAXBException`。以下是使用 `--add-modules` 使该模块可用于编译的方法：

```
$ javac --classpath ${jars} --add-modules java.xml.bind -d ${output-dir} ${source-files}
```

当代码编译和打包时，你需要再次添加模块以执行：

```
$ java --classpath ${jars} --add-modules java.xml.bind ${main-class}
```

如果你依赖于几个 JEE API，添加 `java.se.ee` 模块可能比添加每个单独的模块更容易。它使得所有六个 EE 模块都可用，这简化了一些事情。（它是如何使它们可用的？请阅读第 11.1.5 节中关于聚合模块的内容。）

> **重要信息**：而不是使用 `--add-modules`，我强烈建议认真考虑将所需的 API 的第三方实现作为常规项目依赖项添加。第 9.1.4 节讨论了使用命令行选项的缺点，所以在走这条路之前请务必阅读。而且由于 Java 11 中移除了 JEE 模块，迟早你需要第三方实现。

手动添加 JEE 模块的努力仅适用于非模块化代码。一旦它被模块化，EE 模块就不再特殊：你可以像要求任何其他模块一样要求它们，并且它们将像任何其他模块一样被解析——至少，直到它们被移除。

> **第三方 JEE 实现**
>
> 比较和讨论各种 JEE API 的第三方实现将远离模块系统，因此在这里我不会这么做。有关替代方案列表，请参阅 JEP 320 ([`openjdk.java.net/jeps/320`](http://openjdk.java.net/jeps/320)) 或 Stack Overflow ([`mng.bz/0p29`](http://mng.bz/0p29))。

### 6.1.3 添加 JEE 模块的第三方实现

可能你一直在使用受支持的规范覆盖机制来更新规范和独立技术。在这种情况下，你可能想知道在模块时代它发生了什么。正如你可能猜到的，它被移除并由新事物取代了。

编译器和运行时都提供了 `--upgrade-module-path` 选项，该选项接受一个目录列表，格式类似于模块路径的格式。当模块系统创建模块图时，它会搜索这些目录以查找工件，并使用它们来替换可升级的模块。六个 JEE 模块始终可升级：

- `java.activation`
- `java.corba`
- `java.transaction`
- `java.xml.bind`
- `java.xml.ws`
- `java.xml.ws.annotation`

JDK 供应商可能会使更多模块可升级。例如，在 Oracle JDK 上，这适用于 `java.jnlp`。此外，使用 `jlink` 链接到映像中的应用程序模块始终可升级——有关更多信息，请参阅第 14.2.1 节。

升级模块路径上的 JAR 文件不必是模块化的。如果它们缺少模块描述符，它们将被转换为自动模块（请参阅第 8.3 节），并且仍然可以替换 Java 模块。

## 6.2 将其转换为 `URLClassLoader`

在 Java 9 或更高版本上运行项目时，您可能会遇到类似于以下示例中的类转换异常。在这里，JVM 抱怨它无法将 `jdk.internal.loader.ClassLoaders.AppClassLoader` 实例转换为 `URLClassLoader`：

```
> 线程 "main" 中发生异常 java.lang.ClassCastException:
> java.base/jdk.internal.loader.ClassLoaders$AppClassLoader ①
> 无法转换为 java.base/java.net.URLClassLoader ②
> 在 monitor.Main.getClassPathContent(Main.java:46)
> 在 monitor.Main.main(Main.java:28)
```

> ① `getClass()` 返回的类加载器是 `AppClassLoader`。
>
> ② `AppClassLoader` 不扩展 `URLClassLoader`，因此转换失败。

这是什么新类型，为什么它会破坏代码？让我们找出答案！在这个过程中，您将了解 Java 9 如何更改类加载行为以改进启动性能。因此，即使您的项目没有遇到这个问题，这也是加深 Java 知识的好机会。

### 6.2.1 应用程序类加载器，过去和现在

在所有 Java 版本中，应用程序类加载器（通常称为系统类加载器）是 JVM 用于运行应用程序的三个类加载器之一。它加载不需要任何特殊权限的 JDK 类以及所有应用程序类（除非应用程序使用自己的类加载器，在这种情况下，以下内容不适用）。

