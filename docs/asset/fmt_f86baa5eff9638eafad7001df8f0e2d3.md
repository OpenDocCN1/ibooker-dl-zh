
你是一位 Java 开发者。更新以下由三个井号分隔的代码，并添加 Javadoc 格式的代码注释来解释代码的工作原理。在输出完整的带注释代码之前，请检查代码的任何部分都没有被更改。

我们随后提供使用分隔符策略的代码：

```
###
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Timesheet {

    private ConcurrentHashMap<String, Long> timesheets = 
    ➥new ConcurrentHashMap<>();

    public boolean submitTimesheet(String projectName, long duration) {
        if (projectName != null && duration > 0) {
                    projectName = projectName.toLowerCase();

            timesheets.put(projectName, 
            ➥timesheets.getOrDefault(projectName, 0L) + duration);
            return true;
        } else {
            return false;
        }
    }

    public Collection<Long> getTimesheets() {
        return timesheets.values();
    }

    public long getTotalTimesheetHours(String projectName) {
        return timesheets.getOrDefault(projectName.toLowerCase(), 0L);
    }

}
###
```

将此提示与我们的示例代码一起发送到 ChatGPT 产生：

```
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
/**
 * Represents a timesheet management system for tracking hours spent on
 different projects.
 */
public class Timesheet {

    // A concurrent hash map to store project names and their
    // corresponding total hours
    private ConcurrentHashMap<String, Long> timesheets = new
 ConcurrentHashMap<>();

    /**
     * Submits a timesheet entry for a project.
     *
     * @param projectName The name of the project.
     * @param duration    The duration in hours to be submitted.
     * @return true if the submission was successful, false otherwise.
     */
    public boolean submitTimesheet(String projectName, long duration) {
        if (projectName != null && duration > 0) {
            // Normalize the project name to lowercase for consistency
            projectName = projectName.toLowerCase();

            // Update the total hours for the project or add a new 
            // entry if it doesn't exist
            timesheets.put(projectName, 
timesheets.getOrDefault(projectName, 0L) + duration);
            return true;
        } else {
            return false;
        }
    }

    /**
     * Retrieves a collection of all total timesheet hours for all
     * projects.
     *
     * @return A collection of total hours for all projects.
     */
    public Collection<Long> getTimesheets() {
        return timesheets.values();
    }

    /**
     * Retrieves the total timesheet hours for a specific project.
     *
     * @param projectName The name of the project.
     * @return The total hours spent on the specified project, or 0 
     * if the project is not found.
     */
    public long getTotalTimesheetHours(String projectName) {
        // Normalize the project name to lowercase for consistency 
        // and retrieve the total hours
        return timesheets.getOrDefault(projectName.toLowerCase(), 0L);
    }
}
```

使用这个提示，我们提高了某些注释的质量，并创建了 Javadoc 友好的注释，这些注释可以用来为外部用户记录我们的代码。现在我们只需要调整和编辑我们认为合适的地方。

### 4.5.2 生成发布说明

正如我们可以使用大型语言模型（LLM）来修改我们的代码以添加注释一样，我们也可以使用 LLM 将我们的代码库转换成可以被团队内外其他人消费的发布说明。为此，我们创建了一个包含以下细节的新提示；我们检查解决方案策略以确保发布说明可读：

你是一位 Java 开发者。将以下由三个井号分隔的代码转换为以项目符号形式呈现的发布说明，描述代码的工作原理。在输出之前，请检查发布说明是否可以被没有使用过之前创建的代码的人阅读。

我们添加了一个额外的指令来减少发布说明的冗长性：

| |   *   仅输出发布说明。 |

然后我们使用分隔符策略来指明代码的位置：

```
###
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Timesheet {

    private ConcurrentHashMap<String, Long> timesheets = 
    ➥new ConcurrentHashMap<>();

    public boolean submitTimesheet(String projectName, long duration) {
        if (projectName != null && duration > 0) {
                    projectName = projectName.toLowerCase();

            timesheets.put(projectName,
            ➥timesheets.getOrDefault(projectName, 0L) + duration);
            return true;
        } else {
            return false;
        }
    }

    public Collection<Long> getTimesheets() {
        return timesheets.values();
    }

    public long getTotalTimesheetHours(String projectName) {
        return timesheets.getOrDefault(projectName.toLowerCase(), 0L);
    }

}
###
```

该提示遵循与之前针对代码注释的提示相似的格式，但这次，我们不是要求它向我们的代码中注入注释，而是指示将代码转换为自然语言的发布说明。例如，将包含示例代码的提示发送到 ChatGPT 返回：

发布说明：
*   该代码定义了一个名为 `Timesheet` 的 Java 类，用于管理时间表数据。
*   它使用 `ConcurrentHashMap` 来存储时间表条目，其中项目名称（小写）与其相应的持续时间（小时）相关联。
*   `submitTimesheet` 方法允许用户提交时间表条目。它需要一个项目名称和正持续时间作为参数。如果满足这些条件，它将在映射中存储与项目名称关联的持续时间。如果不满足，则返回 false。
*   `getTimesheets` 方法返回所有记录的时间表持续时间集合。
*   `getTotalTimesheetHours` 方法允许用户检索特定项目名称的总持续时间（不区分大小写）。如果找不到项目名称，则返回 0。注意：在使用 `submitTimesheet` 方法时，请确保提供小写的项目名称，以避免在检索总时间表小时数时出现大小写敏感问题。

我们得到的结果与我们的原始代码完全不同的格式，原始代码试图详细说明我们的代码库做了什么。审查输出，它确实感觉相当技术性，这是有道理的，因为我们要求对一小段代码提供详细的说明。然而，即使这些发布说明也可能有用，因为我们可以根据早期提示的风险分析来调整提示。我们使用分隔符来指示发布说明的位置：

您是一位软件测试专家。分析由三重哈希分隔的发布说明，并识别可能影响代码的风险。如果没有提供代码，请回答“没有找到风险。”

我们随后提供有关质量特性和输出结构的说明：

*   风险必须与质量特性相关：性能、安全性和互操作性
*   每个质量特性都可以识别出多个风险
*   以“风险 - 质量特性 - 说明”的格式输出已识别的风险

少样本提示允许我们给出输出结构的示例：

示例：溢出 - 安全性 - 将大整数作为参数添加可能会导致方法触发缓冲区溢出。

发布说明位于指定的分隔符内：

```
### 发布说明：
*   代码定义了一个名为 `Timesheet` 的 Java 类，用于管理考勤数据。
*   它使用 `ConcurrentHashMap` 存储考勤条目，其中项目名称（小写）与其相应的持续时间（小时）相关联。
*   `submitTimesheet` 方法允许用户提交考勤条目。它需要项目名称和正数持续时间作为参数。如果满足这些条件，它将在映射中存储与项目名称关联的持续时间。如果不满足，则返回 false。
*   `getTimesheets` 方法返回所有记录的考勤持续时间集合。
*   `getTotalTimesheetHours` 方法允许用户检索特定项目名称的总持续时间（不区分大小写）。如果找不到项目名称，则返回 0。注意：在使用 `submitTimesheet` 方法时，请确保提供小写项目名称，以避免在检索总考勤小时数时出现大小写敏感问题。
###
```

使用此提示在 ChatGPT 返回以下建议风险：

*   并发修改 - 性能 - 使用 `ConcurrentHashMap` 可能不适合高并发访问，如果多个用户同时频繁修改考勤条目，可能会导致性能瓶颈。
*   输入验证不足 - 安全 - `submitTimesheet` 方法缺少适当的输入验证。它只检查正数持续时间，但不验证项目名称，使其容易受到注入攻击或其他恶意输入的攻击。
*   大小写敏感 - 互操作性 - 需要使用小写字母提供项目名称的要求，在集成期望不区分大小写输入或具有不同命名约定的系统时可能导致互操作性问题。
*   无身份验证/授权 - 安全 - 发布说明中没有提及任何身份验证或授权机制，如果未经授权的用户可以提交或访问考勤数据，可能会带来安全风险。
*   数据暴露 - 安全 - `getTimesheets` 方法提供了对所有记录的考勤持续时间的访问，如果不适当限制或验证，可能会暴露敏感信息。

当然，如果我们可以直接要求大型语言模型（LLM）根据代码直接提出风险，这可能是一个不必要的步骤，但它确实展示了 LLM 将数据从一种格式转换为另一种格式的强大能力。

## 4.6 与代码助手保持平衡

对于像 Copilot 和 ChatGPT 这样的工具所提供的价值的看法取决于个人。对于一些人来说，这意味着开发和许多职位的自动化，而对于另一些人来说，它只是一个随机选择建议代码的高级工具。这一章所展示的是，它们的价值和用途介于这两个极端之间。

这些模型在 Stack Overflow 和 GitHub 存储库的大量数据上训练，在建议生产代码和测试代码方面非常复杂。但它们仍然需要我们人类的指导——这种指导是由我们与利益相关者沟通、分析需求和设计实现的能力所引导的。我们能在多大程度上在开发中使用 AI 工具，取决于我们磨练互补技能的能力，这可以通过图 4.2 所示的影响区域模型来总结。

平衡的方法可以帮助我们更快地交付功能，但仍然确保我们构建的质量。因此，我们的目标是保持这种平衡，在可能需要依赖我们自身能力或工具特性的情况下。有时，代码助手可能无法建议正确的实现，我们需要承担责任。这给了我们更多的控制权，但牺牲了速度。在其他时候，我们可以依赖代码助手工具通过单元检查或对话来参考大量数据，以提出新的设计想法。然而，我们希望确保我们保持 TDD 循环专注于设计，而不是测试覆盖率。单元检查过多，我们就会失去对设计的关注，最终陷入检查框子的活动中。

![图片](img/CH04_F02_Winteringham2.png)

图 4.2：影响区域模型，更新以展示人类技能和代码助手工具的能力

## 摘要

*   目前市场上大多数生成式 AI 工具都依赖于使用从互联网上抓取的大量数据训练的 LLMs。
*   LLMs 是复杂的算法，它们通过对我们的请求进行统计分析，以确定它们应该响应什么输出。
*   Copilot 是一个使用 OpenAI GPT-4 的编码助手工具，它是在 GitHub 上存储的代码上训练的。
*   Copilot 在 IDE 中工作，并将你的代码作为提示来建议接下来应该添加到测试代码和生产代码中的内容。
*   如 Copilot 之类的工具可以很好地与 TDD 的红色/绿色/重构循环一起工作，帮助我们快速创建单元检查和生产代码。
*   为了帮助 Copilot 返回有价值的代码，我们需要通过提示来引导它。
*   AI 代码助手的成功取决于我们对自身能力和代码助手工具特性的理解。
*   我们主导与工具主导的设计之间存在推拉关系。
*   当平衡从人类转向工具主导时，我们必须意识到其中的权衡。
