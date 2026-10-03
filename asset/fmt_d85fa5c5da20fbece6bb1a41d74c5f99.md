
| `--compress, -c` | 共享字符串字面量，并压缩`lib/modules`（插件） | 14.4.2 |
| `--exclude-files, --exclude-resources` | 排除指定的文件和资源（插件） | 14.4.2 |
| `--generate-jli-classes` | 预生成方法句柄（插件） | 14.4.3 |
| `--include-locales` | 从`jdk.localedata`（插件）中删除除指定区域设置之外的所有区域设置 | 14.4.2 |
| `--launcher` | 在`bin`中为应用程序生成原生启动器脚本 | 14.2.2 |
| `--list-plugins` | 列出可用的插件 | 14.4.1 |
| `--module-path, -p` | 指定查找平台和应用模块的位置 | 14.1.1 |
| `--order-resources` | 在`lib/modules`（插件）中排序资源 | 14.4.1 |
| `--output` | 在指定位置生成镜像 | 14.1.1 |
| `--strip-debug` | 从镜像的字节码中删除调试符号（插件） | 14.4.2 |
| `--suggest-providers` | 列出指定服务的可观察提供者 | 14.1.3 |

## 摘要

+ 命令行工具`jlink`从选定的平台模块创建运行时镜像（使用`jdeps`确定应用程序需要哪些模块）。为了从中受益，应用程序需要运行在 Java 9+ 上，但不需要模块化。
+ 一旦应用程序及其依赖项已完全模块化（不使用自动模块），`jlink`就可以使用它创建应用程序镜像，包括应用程序的模块。
+ 所有对`jlink`的调用都需要指定以下内容：
  + 查找模块的位置（包括平台模块），使用`--module-path`
  + 解析的根模块，使用`--add-modules`
  + 结果镜像的输出目录，使用`--output`
+ 注意`jlink`如何解析模块：
  + 默认情况下，服务不受限制。
  + 带有`requires static`的可选依赖项不会解析。
  + 不允许自动模块。
  + 确保使用`--add-modules`或`--bind-services`单独添加所需的服务提供者或可选依赖项，或者绑定所有提供者。
  + 注意可能没有意识到就隐式依赖的平台服务。一些候选者包括字符集（`jdk.charsets`）、区域设置（`jdk.localedata`）、Zip 文件系统（`jdk.zipfs`）和安全提供者（各种模块）。
+ 由`jlink`生成的运行时镜像：
  + 与使用`--module-path`选择平台模块构建的操作系统绑定
  + 与 JDK 和 JRE 具有相同的目录结构
  + 将平台模块和应用程序模块（统称为系统模块）融合到`lib/modules`
  + 仅包含包含所需模块的二进制文件（在`bin`中）
  + 要启动应用程序镜像，可以使用`bin/java --module ${initial-module}`（不需要模块路径，因为系统模块会自动解析）或使用`--launcher ${name}=${module}/${main-class}`创建的启动器。
  + 使用应用程序镜像，可以使用模块路径添加额外的模块（尤其是提供服务的模块）。具有与系统模块相同名称的模块路径上的模块将被忽略。
+ 当您无法轻松用新版本替换它们时，仔细评估交付应用程序镜像的安全、性能和稳定性影响。
+ 各种`jlink`选项，这些选项激活插件，提供减少图像大小（例如，`--compress`，`--exclude-files`，`--exclude-resource`，`--include-locales`和`--strip-debug`）或提高性能（主要是启动时间；`--class-for-name`，`--generate-jli-classes`和`--order-resources`）的方法。未来可能会有更多；这个领域仍处于早期阶段。
+ `jlink`插件 API 尚未标准化，以简化其在早期阶段的演变，这使得开发和使用第三方插件变得更加困难。

## 15 将各个部分组合在一起

### 本章涵盖

+ ServiceMonitor 的铃声和哨声版本
+ 是否使用模块
+ 一个理想的模块可能的样子
+ 保持模块声明干净
+ 将模块系统与构建工具、OSGi 和微服务进行比较

现在我们已经涵盖了有关模块系统几乎所有的知识，是时候总结一下了。在本章的最后，我想将所有这些联系起来，并给出一些关于创建出色的模块化应用程序的建议。

第一步是向您展示如何通过将本书中讨论的各个功能的大部分应用于 ServiceMonitor 应用程序（第 15.1 节）来展示这些功能如何结合在一起。然后，我将深入探讨一些更一般的问题，这些问题将帮助您决定是否创建模块，在创建模块时应追求什么，以及如何仔细地演进模块声明，使它们保持干净（第 15.2 节）。我将通过回顾围绕模块系统的技术格局（第 15.3 节）以及我对 Java 模块生态系统的愿景（第 15.4 节）来结束。

### 15.1 向 ServiceMonitor 添加铃声和哨声

第二章展示了 ServiceMonitor 应用程序的结构。在第 2.2 节中，您创建了只使用简单的`requires`和`exports`指令的模块。从那时起，我们不仅详细讨论了这些内容，还探索了模块系统的更高级功能。我们逐一研究了它们，但现在我想将它们全部结合起来。

要欣赏 ServiceMonitor 应用程序的全部辉煌，请查看存储库的`features-combined`分支。以下列表包含 ServiceMonitor 中所有模块的声明。

> **列表 15.1 使用本书中介绍的先进功能的 ServiceMonitor**

```
module monitor.observer {
    exports monitor.observer;
    exports monitor.observer.utils ① to monitor.observer.alpha, monitor.observer.beta;
}

module monitor.observer.alpha {
    requires monitor.observer;
    provides monitor.observer.ServiceObserverFactory ② with monitor.observer.alpha.AlphaServiceObserverFactory;
}

// [...] ③

module monitor.statistics {
    requires transitive monitor.observer; ④
    requires static stats.fancy; ⑤
    exports monitor.statistics;
}

module stats.fancy {
    exports stats.fancy;
}

module monitor.persistence {
    requires transitive monitor.statistics; ④
    requires hibernate.jpa; ⑥
    exports monitor.persistence;
    opens monitor.persistence.entity; ⑦
}

module monitor.rest {
    requires transitive monitor.statistics; ④
    requires spark.core; ⑥
    exports monitor.rest;
}

module monitor {
    requires monitor.observer;
    requires monitor.statistics;
    requires monitor.persistence;
    requires monitor.rest;
    uses monitor.observer.ServiceObserverFactory; ②
}
```

> ① `monitor.observer.utils`主要针对观察者实现，因此它仅导出给（一些）它们（参见第 15.1.2 节）。
> ② 观察者 API 的消费者（`monitor`）和实现（例如，`monitor.observer.alpha`）通过服务解耦（参见第 15.1.3 节）。
> ③ `monitor.observer.beta`和`monitor.observer.gamma`在此处未显示；它们看起来就像`monitor.observer.alpha`。
> ④ 一些模块在其 API 中公开了另一个模块的类型，并且没有该模块就无法使用，因此它们暗示了其可读性（参见第 15.1.1 节）。
> ⑤ `stats.fancy`并非在每个部署中都存在，`monitor.statistics`通过将其对该模块的依赖标记为可选来反映这一点（参见第 15.1.1 节）。
> ⑥ ServiceMonitor 使用的 Hibernate 和 Spark 版本都没有模块化，因此`hibernate.jpa`和`spark.core`是自动模块（参见第 15.1.5 节）。
> ⑦ `monitor.persistence`将其持久化实体包含的包打开以供反射（参见第 15.1.2 节）。

如果您将此列表与列表 2.2 进行比较或查看图 15.1，您会发现 ServiceMonitor 的基本结构基本保持不变。但仔细观察，您会发现许多改进。让我们逐一过目。

![图片 1](img/00016.jpeg) ![图片 2](img/00073.jpeg)

**图 15.1 根据功能使用情况比较 ServiceMonitor 应用的模块图。** 第一个变体仅使用普通的`exports`和`requires`指令（a），而第二个变体充分利用了精炼的依赖关系、导出以及服务（b）。（基本变体已扩展，包括与高级变体相同的模块和包。）

#### 15.1.1 多样化的依赖关系

容易发现的一个变化是`requires transitive`和`requires optional`指令。尽管在大多数情况下，普通的`requires`指令是正确的选择，但相当一部分依赖关系要复杂一些。

最明显的情况是可选依赖，其中某个模块使用来自另一个模块的类型，因此需要针对它进行编译，但依赖关系在运行时可能仍然不存在。这正是`monitor.statistics`和`stats.fancy`的情况，因此使用`requires static`指令建立了依赖关系。

模块系统将在编译`monitor.statistics`时强制`stats.fancy`的存在（这是有意义的，因为否则编译会失败），如果后者进入了模块图（这也是有意义的，因为否则`monitor.statistics`无法访问`stats.fancy`的类型），则将从`monitor.statistics`添加一个读取边到`stats.fancy`。但`stats.fancy`可能不会进入模块图，在这种情况下，`monitor.statistics`必须处理其不存在的情况。

> **列表 15.2 检查可选依赖`stats.fancy`是否存在**

```
private static boolean checkFancyStats() {
    boolean isFancyAvailable = isModulePresent("stats.fancy");
    String message = "Module 'stats.fancy' is" + (isFancyAvailable ? " " : " not ") + "available.";
    System.out.println(message);
    return isFancyAvailable;
}

private static boolean isModulePresent(String moduleName) {
    return Statistician.class
            .getModule()
            .getLayer()
            .findModule(moduleName)
            .isPresent();
}
```

可选依赖关系在 11.2 节中进行了详细讨论。

另一种情况比可选依赖稍微不明显，但并不少见——甚至可能更常见。例如，`monitor.rest`模块在其公共 API 中就有这个方法：

```
public static MonitorServer create(Supplier<Statistics> statistics) {
    return new MonitorServer(statistics);
}
```

但`Statistics`来自`monitor.statistics`，因此任何使用`rest`的模块都需要读取`statistics`，否则它无法访问`Statistics`，因此无法创建`MonitorServer`。换句话说，`rest`对于不也读取`statistics`的模块来说是无用的。在 ServiceMonitor 应用中，这种情况出人意料地经常发生：每个至少需要另一个模块并导出包的模块最终都会处于这种情况。

这在野外发生的频率要高得多，而且只有那么频繁，是因为这些模块非常小，几乎所有的代码都是公开的 API——如果它们不在自己的 API 中不断暴露其依赖项的类型，那将令人惊讶。所以尽管在实践中这种情况发生的频率较低，但你仍然可以期望每天都会看到这种情况——在 JDK 中，大约 20%的依赖项被暴露。

为了不让用户猜测他们需要显式要求的其他模块，这既麻烦又使模块声明膨胀，模块系统提供了`requires transitive`。因为`rest` `requires transitive`统计信息，任何读取`rest`的模块也会读取统计信息，因此`rest`的用户免去了猜测的麻烦。隐含的可读性在第 11.1 节中进行了详细讨论。

#### 15.1.2 降低可见性

与 2.2 节中应用程序原始版本相比的另一个变化是，它的模块更加努力地减少它们的 API 表面积。更新的模块使用了相当少的普通`exports`指令：

+ 多亏了服务，观察者不再需要导出它们的实现。
+ 通过使用限定导出，`monitor.observer.utils`包在`monitor.observer`中只能被选定的模块访问。
+ `monitor.persistence`打开其实体包而不是导出它，因此仅在运行时使其可用。

这些变化减少了任何随机模块可以轻松访问的代码量，这意味着开发者可以在模块内部更改更多代码，而无需担心对下游消费者的影响。以这种方式减少 API 表面积对于框架和库的维护性来说是一个福音，但具有许多模块的大型应用程序也可以从中受益。第 11.3 节介绍了限定导出，第 12.2 节探讨了公开包。

#### 15.1.3 通过服务解耦

与 2.2 节相比，模块图的结构性变化仅在于`monitor`不再直接依赖于观察者实现。相反，它只依赖于提供 API 的模块`monitor.observer`，并使用`ServiceObserverFactory`作为服务。所有三个实现模块都提供该服务及其特定实现，模块系统连接了这两者。

这不仅仅是一个美学上的改进。多亏了服务，可以在启动时配置应用程序行为的某些方面——它可以观察哪些类型的服务。可以通过添加或删除提供该服务的模块来添加新实现和删除过时的实现——无需对`monitor`进行任何更改，因此可以使用相同的工件而无需重新构建它们。要了解有关服务的所有信息，请参阅第十章。

#### 15.1.4 在运行时以层叠方式加载代码

尽管服务允许我们在启动时定义应用程序的行为，但我们甚至更进一步。这并没有在模块声明中体现出来，但通过启用监控模块创建新层，我们使得应用程序能够在运行时观察那些在启动时甚至没有`ServiceObserver`实现的服务的功能。按需，监控器将创建一个新的模块图，并使用新的类加载器加载额外的类，并更新其观察者列表。

> **列表 15.3 使用为这些路径上的模块创建的图创建新层**

```
private static ModuleLayer createLayer(Path[] modulePaths) {
    Configuration configuration = createConfiguration(modulePaths);
    ClassLoader thisLoader = getThisLoader();
    return getThisLayer()
            .defineModulesWithOneLoader(configuration, thisLoader);
}

private static Configuration createConfiguration(Path[] modulePaths) {
    return getThisLayer()
            .configuration()
            .resolveAndBind(
                    ModuleFinder.of(),
                    ModuleFinder.of(modulePaths),
                    Collections.emptyList()
            );
}
```

这种行为对于不经常重新部署且重启不便的应用程序尤其有趣。复杂的桌面应用程序可以想到，但运行在客户场所并需要可理解配置的 Web 后端也可能符合条件。关于层是什么以及如何创建它们的讨论，请参阅 12.4 节。
