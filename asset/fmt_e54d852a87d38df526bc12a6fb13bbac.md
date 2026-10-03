
## 12.3.4 使用反射修改模块属性

除了分析模块属性外，您还可以使用`Module`通过调用这些方法来修改它们：

- `addExports`将包导出到模块。
- `addOpens`向模块公开一个包。
- `addReads`允许模块读取另一个模块。
- `addUses`使模块使用一个服务。

在查看这些内容时，您可能会 wonder 为什么可以导出或公开模块的包。这难道不是与强封装相矛盾吗？我们不是在第 12.2 节中讨论了模块所有者必须做什么来为反射做准备，因为反射代码不能破坏吗？

> **重要的信息：** 这些方法是调用者敏感的，这意味着它们根据调用它们的代码表现不同。为了成功调用，它必须来自正在修改的模块内部或来自未命名的模块。否则，它将失败并抛出`IllegalCallerException`。

以下代码作为示例：

```
public boolean openJavaLangTo(Module module) {
    Module base = Object.class.getModule();
    base.addOpens("java.lang", module);
    return base.isOpen("java.lang", module);
}
```

如果将其复制到从类路径执行的`main`方法中（因此它在未命名的模块中运行），这将正常工作，并且方法返回`true`。另一方面，如果它从任何命名模块（以下示例中的`open.up`）内部运行，它将失败：

```
> Exception in thread "main" java.lang.IllegalCallerException:
> java.lang is not open to module open.up
> at java.base/java.lang.Module.addOpens(Module.java:751)
> at open.up/open.up.Main.openJavaLangTo(Main.java:18)
> at open.up/open.up.Main.main(Main.java:14)
```

您可以通过将代码注入它修改的模块（即`java.base`）并使用`--patch-module`（参见第 7.2.4 节）来使其（再次）工作：

```
$ java --patch-module java.base=open.up.jar --module java.base/open.up.Main
> WARNING: module-info.class ignored in patch: open.up.jar
> true
```

这样就完成了：最后的`true`是从带有任意平台模块的`openJavaLangTo`调用返回的值。

动态修改您自己的模块属性不是您会经常做的事情，即使您正在开发基于反射的框架。那么，为什么我要告诉您所有这些？因为正如您将在下一节中看到的那样，这里隐藏了一个有趣的细节：在特定情况下，您可以打开其他模块的包。

## 12.3.5 前向公开包

我说过只有模块可以通过`Module:addOpens`打开其包中的一个，但这并不完全正确。如果一个模块的包已经对一组其他模块打开，那么所有这些模块也可以打开该包。换句话说，具有对包进行反射访问的模块可以将该包打开给其他模块。这意味着什么呢？

再次思考 JPA。你可能在第 12.2.2 节中退缩了，因为看起来你需要无条件地打开一个包或者打开实际进行反射的模块，因为在 JPA 的情况下，这意味着如下所示：

```
module monitor.persistence {
    requires hibernate.jpa;
    requires monitor.statistics;

    exports monitor.persistence;
    // 假设 Hibernate 是一个显式模块，打开 monitor.persistence.entity 到 hibernate.core;
}
```

不打开到 JPA 而不是特定的实现不是更好吗？这正是通过启用具有对包进行反射访问的模块打开包给其他模块的功能所实现的！这样，JPA 的引导代码就可以打开所有包给 Hibernate，即使那些只有反射访问的包。

尽管只有模块可以添加包导出、读取边和服务使用，但打开包的规则已经放宽，所有被打开包的模块都可以将其打开给其他模块。对于基于反射的框架要利用这一点，它们当然必须了解模块系统并更新它们的代码。在 JEE 技术的情况下，这仍然可能需要一段时间，除非 Eclipse 为 Jakarte EE 采用更快的发布周期（从 Java SE 8 到 Java EE 8 花费了超过三年时间）。

现在我们已经解决了如何对单个模块进行反射、分析和修改的问题，我们可以将其提升到下一个层次，或者说在接下来的章节中，我们将看到的工作层，并处理整个模块图。

## 12.4 使用层动态创建模块图

在第 12.3 节中，我们关注了单个模块：如何对模块代码进行反射以及如何分析和修改单个模块的属性。在本节中，我们扩大了我们的范围，并查看整个模块图。

到目前为止，我们将模块图的创建留给了编译器或 JVM，它们在开始工作之前生成它们。从那时起，图几乎是一个不可变的实体，没有提供添加或删除模块的方法。

虽然这对许多常规应用程序来说很好，但还有一些需要更多灵活性的应用程序。想想应用服务器或基于插件的程序。它们需要一个动态机制，允许它们在运行时加载和卸载类。

例如，假设`ServiceMonitor`应用程序提供了一个端点或图形界面，用户可以通过它指定需要观察的附加服务。这可以通过实例化适当的`ServiceObserver`实现来完成，但如果该实现来自启动时未知的模块呢？那么（及其依赖项）就必须在运行时动态加载。

在模块系统之前，此类容器应用程序使用裸类加载器进行动态加载和卸载，但它们是否也能像编译器和 JVM 一样，将其提升到更高的抽象层次，并操作模块呢？幸运的是，模块系统通过引入层的概念实现了这一点。首先，你需要了解层，包括那些你一直不知道的层（第 12.4.1 节）。下一步是在运行时动态创建自己的层之前分析层（第 12.4.2 节）。

注意，处理层的代码甚至比使用反射 API 更不常见。这里有一个简单的试金石：如果你从未实例化过类加载器，你不太可能很快使用层。因此，本节为你提供了地形图，让你知道自己的位置，但不会深入细节。尽管如此，你可能会看到一些你不知道可以实现的事情，并最终得到一些新想法。

### 12.4.1 层是什么？

> **定义：模块层**
>
> 模块层包含一个名为模块的完全解析的图以及用于加载模块类的类加载器（们）。每个类加载器都与一个未命名的模块相关联（可以通过`ClassLoader::getUnnamedModule`访问）。层还引用一个或多个父层——层中的模块可以读取祖先层中的模块，但反之则不行。

我们到目前为止讨论的所有关于模块的解析和关系都发生在单个模块图中。使用层，可以堆叠尽可能多的图，因此，从概念上讲，层为二维模块图的概念添加了第三个维度。父层在创建层时定义，之后不能更改，因此无法创建循环层。图 12.2 显示了带有层的模块图。

![图](img/00121.jpeg)

![图 12.2](img/part0027.html#filepos1574970) 使用层，模块图可以堆叠，为你的应用程序的心理模型添加第三个维度。因为它们不共享类加载器，所以层之间很好地隔离。（像每个好的计算机科学图一样，这个图可能看起来是颠倒的。父层位于其子层下方，因为这样可以保持包含平台模块的层在底部。）

对于`ServiceMonitor`来说，这意味着为了动态加载新的观察者实现，它需要创建一个新的层。在我们第 12.4.3 节中讨论这一点之前，让我们更仔细地看看现有的层以及如何分析它们。

所有的模块都包含在一个层中吗？几乎是这样。正如你所看到的，从技术上来说，未命名的模块不是。然后还有所谓的动态模块，它们不必属于一个层，但我在这本书中不会涉及它们。除了这些例外，所有模块都是层的组成部分。

**引导层**

那么，在这本书中通过图形放入的所有应用程序和平台模块呢？它们也应该属于一个层，对吧？

> **定义：引导层**
>
> 的确如此。在启动时，JVM 创建一个初始层，即引导层，它包含根据命令行选项解析的应用程序和平台模块。

引导层没有父层，包含三个类加载器：

- 引导类加载器为它加载的所有类授予所有安全权限，因此 JDK 团队努力最小化它负责的模块；这些是一些核心平台模块，其中最重要的是`java.base`。
- 平台类加载器从所有其他平台模块中加载类；可以通过静态方法`ClassLoader::getPlatformClassLoader`访问它。
- 系统或应用程序类加载器从模块和类路径中加载所有类，这意味着它负责所有应用程序模块；可以通过静态方法`ClassLoader::getSystemClassLoader`访问它。

只有系统类加载器可以访问类路径，因此在这三个加载器中，只有它的未命名模块可能不为空。因此，当第 8.2 节提到未命名模块时，它总是引用系统类加载器的。

正如你在图 12.3 中可以看到的，类加载器不是孤岛：每个类加载器都有一个父类加载器，包括刚刚提到的三个在内的大多数实现，首先会请求父类加载器加载一个类，然后再尝试自己查找。对于三个引导层类加载器，引导层是没有任何父类的祖先，平台委托给引导层，系统委托给平台。因此，系统类加载器可以访问来自引导层和平台加载器的所有应用程序和 JDK 类。

![图片](img/00028.jpeg)

![图 12.3](img/part0027.html#filepos1579881) 引导层中三个类加载器之间的委托

### 12.4.2 分析层

> **定义：ModuleLayer**
>
> 在运行时，层由`java.lang.ModuleLayer`实例表示。它们可以查询层由以下三个部分组成：
>
> - 模块：
>   - 方法`modules()`返回层包含的模块，作为一个`Set<Module>`。
>   - 方法`findModule(String)`在自身层以及所有祖先层中搜索具有指定名称的模块。因为它可能找不到，所以它返回一个`Optional<Module>`。
> - 通过`parents()`方法，层的父级以`List<ModuleLayer>`的形式返回。
> - 通过调用`findLoader(String)`并传递模块的名称，可以确定每个模块的类加载器。
>
> 然后是`configuration`方法，它返回一个`Configuration`实例——有关更多内容，请参阅第 12.4.3 节。

要获取`ModuleLayer`实例，您可以要求任何模块提供它所属的层：

```
Class<?> type = // ... 任何类
ModuleLayer layer = type
    .getModule()
    .getLayer();
```

如果类型来自未命名的模块或不属于层的动态模块，则最后一行返回`null`。如果您想访问引导层，可以调用静态的`ModuleLayer::boot`方法。

那么，从`ModuleLayer`实例中你可以学到什么？毫无疑问，最有趣的方法是`modules()`和`findModule(String)`，因为与`Module`上的方法（参见第 12.3.3 节）一起，它们允许遍历和分析模块图。

**描述模块层**

给定列表 12.7 中的`describe(Module)`方法，这就是如何描述整个层的方法：

```
private static String describe(ModuleLayer layer) {
    return layer
        .modules().stream()
        .map(ThisClass::describe)
        .collect(joining("\n\n"));
}
```

**在层内和层间查找模块**

还可以确定特定模块的存在或不存在，如果对这些模块的依赖是可选的（使用`requires static`；参见第 11.2 节）。在第 11.2.4 节中，我声称实现一个名为`isModulePresent(String)`的方法来做这件事是直接的。这使你将迄今为止学到的关于层的内容付诸实践，所以让我们一步一步来做。

起初这似乎很简单：

```
public boolean isModulePresent(String moduleName) {
    return ModuleLayer
        .boot()
        .findModule(moduleName)
        .isPresent();
}
```

但这仅显示模块是否存在于引导层。如果创建了额外的层，并且模块位于另一个层中怎么办？您可以替换引导层为包含`isModulePresent`的层：

```
public boolean isModulePresent(String moduleName) {
    return searchRootModuleLayer()
        .findModule(moduleName)
        .isPresent();
}

private ModuleLayer searchRootModuleLayer() {
    return this
        .getClass()
        .getModule()
        .getLayer();
}
```

这样，`isModulePresent`会搜索包含自身的层——让我们称它为`search`——以及所有父层。但即使这样也不够好。调用该方法的模块可能位于不同的层，称为`call`，该层将`search`作为祖先。（困惑吗？参见图 12.4。）然后`search`无法查看`call`，因此无法搜索所有可能的模块。不，您需要调用者的模块来使用其层作为搜索的根。

![图片](img/00035.jpeg)

图 12.4 请求查找模块的层只扫描它自己和它的父级（在这个图中，是向下）。所以如果`search`查询它自己的层，它可能会忽略调用层，即启动搜索的模块可以看到的层，从而冒着返回错误结果的风险。这就是为什么查询调用层很重要。

以下列表实现了`getCallerClass`，它使用 Java 9 引入的堆栈遍历 API 确定调用者的类。

> **列表 12.9 新的 API 用于遍历调用栈**

```
private Class<?> getCallerClass() {
    return StackWalker
        .getInstance(RETAIN_CLASS_REFERENCE)
        .walk(stack -> stack
            .filter(frame -> frame.getDeclaringClass() != this.getClass())
            .findFirst()
            .map(StackFrame::getDeclaringClass)
            .orElseThrow(IllegalStateException::new));
}
```

> ① 静态工厂方法用于获取一个`StackWalker`实例，其中每个帧都引用了声明类
>
> ② `StackWalker::walk`期望一个从`Stream<StackFrame>`到任意对象的函数。它创建了一个对堆栈的懒视图，并立即用它调用函数。函数返回的对象随后由`walk`返回。
>
> ③ 你对来自不是这个类的第一个帧感兴趣（这必须是调用者！）；你现在有一个`Optional<StackFrame>`。
>
> ④ 获取那个类
>
> ⑤ 如果没有这样的帧存在，那就奇怪了……

在你的工具箱中有这个，调用者的模块就在你的指尖：

```
public boolean isModulePresent(String moduleName) {
    return searchRootModuleLayer()
        .findModule(moduleName)
        .isPresent();
}

private ModuleLayer searchRootModuleLayer() {
    return getCallerClass()
        .getModule()
        .getLayer();
}
```

分析层到此结束。现在我们终于可以进入最激动人心的部分：通过创建新层将新代码加载到运行中的应用程序中。

### 12.4.3 创建模块层

只有少数用 Java 编写的应用程序需要在运行时动态加载代码。同时，这些往往是更重要的。最著名的可能是 Eclipse，它对插件有很强的关注，但像 WildFly 和 GlassFish 这样的应用服务器也必须同时从一个或多个应用程序中加载代码。如 1.7.3 节所述，OSGi 也能够动态加载和卸载包（它对模块的称呼）。

它们对它们使用的加载插件、应用程序、包和其他运行 JVM 的新片段的机制有相同的基本要求：

- 必须能够在运行时从一组 JAR 文件中启动一个片段。
- 必须允许与加载的片段交互。
- 必须允许不同片段之间的隔离。

在模块系统之前，这是通过类加载器完成的。简要来说，为新的 JAR 文件创建了一个新的类加载器。它委托给另一个类加载器，例如系统类加载器，这使它能够访问运行中的 JVM 中的其他类。尽管每个类（通过其完全限定名标识）在每个类加载器中只能存在一次，但它可以很容易地被多个加载器加载。这隔离了片段，并给每个片段提供了在没有与其他片段冲突的情况下提出其自身依赖的可能性。
