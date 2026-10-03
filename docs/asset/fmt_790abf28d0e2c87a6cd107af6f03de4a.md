

# 第八章：使用 ChatGPT 进行安全的应用程序开发

## 本章涵盖

+ 使用 ChatGPT 进行威胁建模
+ 使用 ChatGPT 培养安全思维
+ 使用 ChatGPT 缓解风险

在不断发展的软件开发领域，安全担忧已经从事后考虑转变为项目设计和实施阶段的重要组成部分。尽管这种提高的关注度，开发者们往往发现难以跟上应用程序安全领域快速变化的步伐。本章提供了对如何将 AI，特别是 ChatGPT，嵌入到应用程序开发过程的各个阶段以增强应用程序安全性的全面理解，为构建更安全的软件应用程序提供了一套新颖的工具。

随着我们深入探讨这个主题，我们将探讨如何将 ChatGPT 集成到 ISAM 应用程序的开发过程的各个阶段，该应用程序是用 FastAPI 编写的 Python。我们将讨论这个 AI 模型如何帮助识别漏洞、参与威胁建模、评估应用程序设计以识别潜在的不安全性、理解和应用安全最佳实践。

本章的目标不是将 ChatGPT 定位为解决所有安全问题的银弹，而是展示其在开发者安全工具包中作为强大工具的潜力。我们将学习如何积极识别和管理威胁，同时考虑到创建不仅功能性强而且安全的软件的整体目标。在这个过程中，我们将探讨建模威胁、在开发生命周期中融入安全、AI 在安全中的作用等众多话题。让我们开始吧！

## 安全不是一个特性

安全的应用程序始于设计。虽然它经常被当作一个特性来对待，但它并不是一个特性。生成式 AI 是可用于评估和改进应用程序安全性的工具，但它们不会取代安全专家的输入，也不会使你成为安全专家。有关设计安全应用程序的更多信息，请参阅丹·伯格·约翰逊、丹尼尔·德古恩和丹尼尔·萨瓦诺所著的《设计安全》（Manning，2019；[www.manning.com/books/secure-by-design](https://www.manning.com/books/secure-by-design)）。

![](img/CH08_F01_Crocker2b.png)

软件生命周期（错误）的心理模型，其中安全被视为一个需要根据项目需要优先考虑和降级的特性，或者作为一个在生命周期某个阶段执行的附加功能。然而，安全是一种需要在所有阶段都放在首位的心态。

## 8.1 使用 ChatGPT 进行威胁建模

威胁建模是一种结构化的方法，它帮助团队理解、优先排序和管理系统中潜在威胁。通过模拟攻击者的心态，威胁建模系统地识别漏洞、评估潜在影响，并确定缓解策略。根植于设计阶段，但与整个软件开发生命周期相关，威胁建模在高级安全策略和实际操作之间发挥着至关重要的桥梁作用。

威胁建模不是一个一次性过程。随着新漏洞的发现，系统和外部环境发生变化，因此你必须重新审视并更新你的威胁模型。

### 8.1.1 为什么它在当今的开发环境中很重要

在我们深入探讨使用 ChatGPT 进行威胁建模之前，我们需要退一步思考，为什么我们首先想要做这件事。在当今的开发环境中，对安全的重视程度越来越高，我们必须关注那些显著改变软件开发、部署和访问方式的因素。随着更多服务数字化，攻击面变得更广。从在线银行到健康记录、电子商务，甚至政府服务，现在都可在网上获取，使其成为潜在的目标。

此外，网络威胁不是静态的。新的漏洞每天都在出现，攻击者不断制定新的方法。随着国家支持的网络攻击、勒索软件和网络间谍活动的兴起，危险从未如此严重。

现代架构比以往任何时候都要复杂，因为应用程序通常使用微服务、第三方 API 和云基础设施。这种复杂性可以引入多个潜在的漏洞点。系统不再是孤立的，而是与其他系统相互连接，产生多米诺效应。一个系统的漏洞可能成为攻击其他系统的跳板。

### 安全漏洞

除了直接的财务影响之外，安全漏洞还可能损害信任，损害公司的声誉，导致法律后果，并导致客户或商业机会的损失。此外，随着欧洲的通用数据保护条例（GDPR）和美国加州消费者隐私法案（CCPA）等法规的实施，组织在保护用户数据方面承担着更大的责任。不遵守规定可能导致重大罚款。

在一个互联、以数字优先的世界里，安全不仅仅是 IT 问题，更是基本的企业需求。确保应用程序从底层开始就考虑安全性可以降低风险和成本，建立信任，并确保系统连续性。

### 8.1.2 ChatGPT 如何帮助进行威胁建模

现在我们已经理解了原因，让我们转向如何利用 ChatGPT 来了解我们周围的网络安全威胁、它们的影响以及潜在的缓解技术。ChatGPT 拥有广泛的网络安全基础知识的知识库；它可以定义标准术语，并以适合你网络安全旅程的任何详细程度向你解释复杂的攻击向量。你可以先要求它解释网络安全原则，什么是 SQL 注入攻击（但不是如何执行它！），或者什么是点击劫持。

作为一种非正式的建模威胁的方法，你可以向 ChatGPT 提出详细假设场景，并询问在这些情况下可能出现的潜在威胁或漏洞。开始时要非常一般化，随着过程的进行进行细化。例如，你可能输入以下提示：

| ![](img/logo-NC.png) | 如果我正在开发一个基于云的电子商务 Web 应用，我应该注意哪些威胁？ |
| --- | --- |

然后深入挖掘，围绕特定威胁进行三角定位：

| ![](img/logo-NC.png) | 攻击者如何在我的电子商务应用中劫持用户的购物车？ |
| --- | --- |

接下来，你可以与 ChatGPT 互动，了解如何评估与各种威胁相关的风险。这可以帮助你确定哪些威胁应该首先解决。在了解一些可能针对你系统的威胁后，你可以与 ChatGPT 讨论潜在的对策、最佳实践和缓解策略。

| ![](img/logo-NC.png) | 我如何评估我的在线服务遭受 DDoS 攻击的风险？ |
| --- | --- |

然后，

| ![](img/logo-NC.png) | 防止跨站脚本攻击的最佳实践是什么？ |
| --- | --- |

你需要定期与 ChatGPT 互动，以刷新你的知识或询问你遇到的新概念或策略。

然而，有一个快速警告：就像往常一样，你必须意识到 ChatGPT 的限制。它没有实时威胁情报或超出其最后更新的知识。对于最新的威胁，请始终咨询最新的资源。尽管 ChatGPT 是一个有价值的工具，但始终与其他权威来源交叉验证其见解。网络安全迅速发展，与多个可信来源保持更新至关重要。在与 ChatGPT 讨论特定威胁后，你可能想查阅来自开放世界应用安全项目（OWASP）、国家标准与技术研究院（NIST）和其他公认网络安全实体等组织的最新文档。

最后，与 ChatGPT 进行互动头脑风暴可以帮助你有效地生成想法，理解复杂的概念，或完善策略，尤其是在像网络安全这样的领域。以下是如何构建和执行此类会议的步骤：

1. 明确说明头脑风暴会议的目标。例如，可以是识别系统中的潜在漏洞、为新的应用程序生成安全措施或讨论提高用户数据保护策略。
2. 通过向 ChatGPT 提供详细背景信息开始会议。如果是关于特定系统或应用程序，请描述其架构、组件、功能以及任何已知的问题或担忧。例如，您可以说：

| ![](img/logo-NC.png) | 我正在开发一个基于 Web 的电子商务平台，使用 Docker 容器进行微服务架构。我们正在寻找潜在的安全威胁。 |
| --- | --- |

根据 ChatGPT 的回应，深入探讨特定领域或关注点。例如，您可能说：

| ![](img/logo-NC.png) | 请告诉我更多关于容器安全最佳实践的信息。 |
| --- | --- |

或询问：

| ![](img/logo-NC.png) | 我如何确保微服务之间的通信安全？ |
| --- | --- |

3. 向 ChatGPT 提出假设情景，并请求反馈或解决方案。这有助于预测潜在的挑战或威胁：

| ![](img/logo-NC.png) | 假设攻击者获得了对其中一个容器的访问权限；应该采取哪些步骤？ |
| --- | --- |

4. 通过扮演魔鬼的代言人，与 ChatGPT 互动。质疑或反驳它提供的想法或建议，以激发进一步的思考并探索不同的角度：

| ![](img/logo-NC.png) | 如果我使用第三方身份验证服务怎么办？这会如何改变安全格局？ |
| --- | --- |

5. 向 ChatGPT 请求具体的步骤或行动项来实施建议的解决方案。例如，您可以询问：

| ![](img/logo-NC.png) | 针对您提到的安全担忧，我应该采取哪些具体步骤来减轻它们？ |
| --- | --- |

注意：随着头脑风暴的进行，记录 ChatGPT 提供的思想、建议和策略。它们在会议后的回顾和实施中将非常有价值。头脑风暴在迭代时最为有效。根据您在一次会议中学到的知识，您可能需要细化问题、调整方法或在下一次会议中探索新的领域。

图 8.1 显示了在头脑风暴会议期间执行的安全反馈循环。

![](img/CH08_F02_Crocker2.png)

图 8.1 与 ChatGPT 进行头脑风暴会议的工作流程

随着您的项目或场景的进展，重新与 ChatGPT 讨论，以考虑任何变化、更新或新的挑战。最近的更新允许您上传您的系统设计文档，并让 ChatGPT 审查该设计以识别潜在威胁和常见漏洞，就像它可以评估代码一样。

通过以这种方式使用 ChatGPT，您可以从中广泛的知识库中受益，并为您的场景获得有价值的反馈和见解。始终记得将建议与相关领域的最新资源和专家进行交叉验证。

### 8.1.3 案例研究：使用 ChatGPT 模拟威胁建模

除了与 ChatGPT 进行场景分析和互动头脑风暴会议之外，您还可能决定应用一种正式的方法，这是网络安全专业人士常用的：STRIDE。STRIDE 是由微软引入的一种威胁建模方法，旨在帮助识别系统或应用程序中的潜在安全威胁。您可以使用 ChatGPT 和 STRIDE 来模拟威胁并相应地进行分类。

### STRIDE

STRIDE 这个缩写代表欺骗、篡改、否认、信息泄露和拒绝服务：

+ *欺骗*—冒充他人或他物。这可能意味着承担用户的身份、设备或甚至系统的身份以获得未经授权的访问。例如，攻击者可能通过欺骗媒体访问控制（MAC）地址来冒充网络上的设备。
+ *篡改*—未经授权修改数据或代码。例如，攻击者可能在中间人攻击中修改传输的数据包。
+ *否认*—恶意（或粗心）用户否认他们执行了特定行为的能力，即使他们确实执行了，也因为缺乏这些行为的证据。例如，用户可能否认下订单或更改某些数据。
+ *信息泄露*—将信息暴露给那些不应有权访问它的人。例如，数据库配置错误可能将敏感用户数据暴露给公众。
+ *拒绝服务（DoS）*—使资源对预期用户不可用或不响应。这可以通过耗尽系统资源或利用特定漏洞来实现。例如，在 DDoS 攻击期间，攻击者可能通过流量淹没一个网站，导致其变慢或完全无响应。
+ *权限提升*—无特权的用户获得了提升的权限，通常允许他们控制整个系统或执行他们未经授权的任务。

在本书的整个过程中，我们一直在开发 ITAM 系统的一部分。作为提醒，该系统用于管理和优化组织的 IT 资产。在本节中，我们将应用 STRIDE 到我们的应用程序中，使用 ChatGPT 协助建模针对该系统的威胁。

让我们从欺骗开始：

| ![](img/logo-NC.png) | 在我们 ITAM 系统的背景下，我们的系统可能如何被欺骗？ |
| --- | --- |

ChatGPT 可能响应的内容可能与以下列表类似。

列表 8.1 ChatGPT 解释欺骗可能如何影响我们的应用程序

```py
MAC Address Spoofing: Most ITAM systems rely on MAC addresses to identify and manage devices on a network. An attacker might spoof the MAC address of an authorized device to gain unauthorized access or to mask a rogue device's presence.
IP Address Spoofing: An attacker could bypass network security measures or divert traffic by masquerading as a trusted IP address. This could lead to unauthorized access to the ITAM system or even denial-of-service attacks.
DNS Spoofing (Cache Poisoning): If the ITAM system relies on domain names for accessing updates or related resources, DNS spoofing can redirect the ITAM server to malicious sites, potentially compromising the system.
Device Identifier Spoofing: Beyond MAC addresses, devices may have other unique identifiers. Spoofing these can allow unauthorized devices to appear as legitimate assets in the ITAM inventory.
User Agent Spoofing: If the ITAM system is accessed via web browsers, attackers can spoof user agent strings to mimic authorized devices or browsers, potentially bypassing certain security checks.
Fake Asset Registration: A malicious actor might introduce fake assets into the system, leading to inaccurate asset tracking, skewed analytics, and potential security loopholes.
Spoofed Update Servers: If your ITAM solution fetches updates from remote servers, attackers could spoof these servers, delivering malicious updates or software.
Phishing & Social Engineering: While not a technical spoofing method per se, attackers can still spoof emails or other communications, pretending to be ITAM administrators or vendors. Unsuspecting employees might then provide sensitive ITAM data.
```

此外，它还可能响应常见的预防或缓解步骤。

列表 8.2 ChatGPT 概述防止此类利用的机制

```py
Network Security: Implement network segmentation, strong firewalls, and Intrusion Detection Systems (IDS) to monitor and block suspicious activities.
Encryption: Encrypt data in transit to protect against man-in-the-middle attacks that could lead to spoofing.
Dynamic Identification: Rotate device identifiers periodically or use dynamic identifiers.
DNS Security: Implement DNS Security Extensions (DNSSEC) to ensure the authenticity of DNS responses.
Authentication and Authorization: Require multi-factor authentication (MFA) for ITAM system access and ensure strict role-based access control.
Regular Audits: Audit the ITAM database frequently to identify discrepancies or rogue assets.
User Education: Regularly train IT personnel and employees about the risks of spoofing and the importance of verifying communications.
```

我们不一一列举 STRIDE 的所有内容，接下来让我们转向否认。这是所有攻击中最不直观的，也是被误解最多的：

| ![](img/logo-NC.png) | 在我们 ITAM 系统的背景下，我们的系统可能受到否认的影响吗？ |
| --- | --- |

ChatGPT 可能在我们系统的背景下对这种风险给出定义。

列表 8.3 ChatGPT 从网络安全的角度定义否认

```py
Repudiation, in a cybersecurity context, refers to the ability of a user (or system) to deny having performed a specific action, even if they did perform it. Regarding ITAM (Information Technology Asset Management) systems, repudiation can pose challenges regarding accountability, traceability, and compliance. Let's explore how repudiation might affect an ITAM system:
```

