
![6](img/#co_deploying_your_api_to_the_cloud_CO1-6)

在步骤 1 到 5 构建容器镜像之后，此步骤设置默认命令，该命令将在使用`docker run`命令从该镜像启动容器时执行。就像你在 Render 部署中所做的那样，你直接调用 Uvicorn 来运行 API。

这些就是您需要定义容器镜像的所有指令。下一步将在您的本地仓库中构建容器镜像。

## 创建一个.dockerignore 文件

就像`.gitignore`排除版本控制中的文件一样，您希望排除目录中的一些文件，使其不在 Docker 镜像中。创建一个名为`.dockerignore`的文件，并包含以下内容：

```py
# This is the .dockerignore file for Chapter 6
.gitignore

README.md

*.DS_Store/

**/__pycache__
```

## 构建容器镜像

要使用此 Dockerfile，请在命令行中输入以下命令：

```py
.../chapter6 (main) $ docker build -t apicontainerimage .
```

这告诉 Docker 使用当前目录中的 Dockerfile 构建一个镜像（`apicontainerimage`）。容器镜像将存储在您的本地 Docker 仓库中。

您应该看到多个步骤正在执行。第一次运行此命令时，这可能需要几分钟。未来的构建将花费更少的时间，因为 Docker 可以缓存未更改的项目。完成时，您应该看到类似以下内容：

```py
[+] Building 12.2s (11/11) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 670B
 => [internal] load .dockerignore
 => => transferring context: 92B
 => [internal] load metadata for docker.io/library/python:3.10-slim
 => [1/6] FROM docker.io/library/python:3.10-slim@sha256:xxxx
 => [internal] load build context
 => => transferring context: 217B
 => CACHED [2/6] WORKDIR /code
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir --upgrade -r requirements.txt
 => [5/6] COPY *.py .
 => [6/6] COPY *.db .
 => exporting to image
 => => exporting layers
 => => writing image sha256:xxxx
 => => naming to docker.io/library/apicontainerimage
```

为了验证镜像创建成功，请输入命令`docker images`来查看您仓库中的镜像。您应该看到类似以下内容：

```py
.../chapter6 (main) $ docker images
REPOSITORY            TAG       IMAGE ID       CREATED          SIZE
apicontainerimage     latest    x9999        1 minute ago     159MB
```

## 在本地运行容器镜像

您使用`docker run`命令基于此镜像运行一个容器。这里有几个选项需要注意。`--publish 80:80`语句将 Docker 容器内部的端口 80（第二个 80）映射到您的本地环境中的端口 80（第一个 80）。`--name apicontainer1`语句设置了您将要使用的容器的名称。这是一个方便的方式来引用正在运行的容器（Docker 也会分配一个镜像 ID）。最后，`apicontainerimage`传递了您在上一步骤中构建的镜像名称。记住，镜像用于运行容器。

执行以下命令：

```py
.../chapter6 (main) $ docker run --publish 80:80 --name apicontainer1
apicontainerimage+
```

您应该看到以下内容：

```py
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```

您将看到消息“您的应用程序在端口 80 上运行可用。”选择“在浏览器中打开”以在 Web 浏览器中运行 API。您应该在窗口中看到健康检查消息，这与您从命令行运行 API 时完全一样。

现在，如果您回到终端窗口，您将看到对 API 的第一个请求的日志消息：

```py
INFO:     1.1.0.1:12345 - "GET / HTTP/1.1" 200 OK
```

恭喜！您的应用程序正在由 Docker 在您定义的容器中运行。要停止应用程序，请按 Ctrl-C。

# 部署到 AWS

AWS 部署将利用您创建的 Docker 容器。您将使用 Amazon Lightsail 服务运行您的应用程序，这是 AWS 中启动起来相对简单的服务之一。

首先，[创建一个新的 AWS 账户](https://oreil.ly/09mZ7) 并安全地存储您的登录凭证。对于这个项目，使用*root 用户*账户是可以接受的，它是通过电子邮件地址识别的账户的全权所有者。然而，您应该启用多因素认证（MFA），按照[AWS 身份和访问管理用户指南](https://oreil.ly/N5Q7e)中的说明操作。

如果您继续使用 AWS，一个最佳实践是创建一个单独的管理账户，并使用它进行常规工作，而不是使用 root 用户。

## 创建一个 Lightsail 容器服务

当你登录到 AWS 控制台后，从搜索栏中选择 Lightsail，然后从左侧菜单中选择容器。你应该会看到一个类似于图 6-7 的页面。点击“创建容器服务”。

![Lightsail 主页](img/haad_0607.png)

###### 图 6-7\. Lightsail 的起始页面

你将被要求完成第一个 Lightsail 服务的相关信息：

+   *区域*：选择离你最近的一个区域以获得最佳性能。

+   *功率*：选择 Nano 或 Micro。

+   *规模*：选择 1。

+   *标识你的服务*：为你的容器选择一个名称。本例中使用`aws-api-container`。

点击“创建容器服务”。创建容器可能需要几分钟时间。完成时，你应该会看到一个状态页面，类似于图 6-8 中的页面。

![Lightsail 实例完成](img/haad_0608.png)

###### 图 6-8\. Lightsail 容器服务创建完成

## 安装 AWS CLI

为了与 AWS 交互，你将使用 AWS CLI。按照 AWS 的说明[安装或更新 AWS CLI 的最新版本](https://oreil.ly/r8tTp)。

运行以下命令以验证安装：

```py
.../chapter6 (main) $ aws --version
aws-cli/2.15.8 Python/3.11.6 Linux/6.2.0-1018-azure exe/x86_64.ubuntu.20
prompt/off
```

## 安装 Amazon Lightsail 容器服务插件

对于 Lightsail，还需要进行额外的安装。按照[安装 Amazon Lightsail 容器服务插件](https://oreil.ly/xMzkI)的说明进行操作。

## 配置您的登录凭证

为了将 AWS CLI 连接到你的 AWS 账户，你需要配置你的身份验证和访问凭证。有多种方法可以做到这一点，正确设置可能需要一些时间。正如本章多次提到的，你应该非常小心地处理云服务中的凭证，确保它们永远不会被提交到源代码仓库。按照[“AWS CLI 的身份验证和访问凭证”](https://oreil.ly/HAXYD)中的说明来配置你的 AWS CLI。

一旦你完成了配置凭证的说明，请通过输入以下命令来验证它们：

```py
.../chapter6 (main) $ aws sts get-caller-identity
```

你应该会收到以下格式的响应：

```py
{
    "UserId": "99999",
    "Account": "999",
    "Arn": "arn:aws:iam::1234:user/username"
}
```

## 将你的容器镜像推送到 Lightsail

接下来，你将使用 AWS CLI 将容器镜像推送到 Lightsail。如果在执行这些说明时遇到任何问题，可以在[“推送、查看和删除 Lightsail 容器服务的容器镜像”页面](https://oreil.ly/lJ0wX)找到更多信息。

为了验证你的 Docker 镜像是否仍然在你之前章节中构建的本地仓库中，请输入以下命令：

```py
.../chapter6 (main) $ docker images
REPOSITORY           TAG       IMAGE ID       CREATED        SIZE
apicontainerimage    latest    aa0366008bec   2 hours ago   159MB
```

你将从本地的 Docker 仓库推送一个镜像到 Lightsail。花点时间理解你将要填写的一些值：

`区域`

这是你设置 Lightsail 服务的 AWS 区域。

`容器服务名称`

如果你在前面的步骤中正确设置了 Lightsail 服务，这应该是`aws-api-container`。

`LocalContainerImageName:ImageTag`

这是你的本地存储库中的名称和标签。在前面的示例中，值将是`apicontainerimage:latest`。

在终端中输入以下命令，用括号中的值替换你的信息：

```py
lightsail push-container-image --region <*Region*>
--service-name <*ContainerServiceName*> --label aws-api --image
<*LocalContainerImageName*>:<*ImageTag*>

```

###### 注意

这些命令可能会在这本书中换行，但你将在终端的单行中输入它们。

这里是一个完整填写命令的示例：

```py
.../chapter6 (main) $ aws lightsail push-container-image --region us-west-2
--service-name aws-api-container --label aws-api --image apicontainerimage:latest
```

几分钟后，你应该看到以下内容：

```py
1234: Pushed
12345: Pushed
Digest: sha256:xxxxx
Image "apicontainerimage:latest" registered.
Refer to this image as ":aws-api-container.aws-api.1" in deployments.
```

返回 Lightsail 控制台登录并查看`aws-api-container`上的镜像选项卡。你应该看到类似于图 6-9 的内容。

![Lightsail 存储的镜像](img/haad_0609.png)

###### 图 6-9\. Lightsail 存储的镜像页面

## 创建 Lightsail 部署

在`aws-api-container`服务的部署选项卡上，选择“创建你的第一个部署”。输入以下值，如图 6-10 图 6-10 所示：

+   *容器名称*：选择一个唯一的容器名称。此示例使用`aws-api-container-1`。

+   *镜像*：选择之前步骤中推送的存储镜像。

![Lightsail 顶级选项](img/haad_0610.png)

###### 图 6-10\. 部署选项（页面顶部）

点击“+ 添加开放端口”并在端口中输入 80。在公共端点部分，点击下拉框并选择你创建的容器。

在这些步骤之后，完成后的选项应如图 6-11 图 6-11 所示。

![Lightsail 顶级选项](img/haad_0611.png)

###### 图 6-11\. 部署选项（页面底部）

点击“保存并部署”。Lightsail 将开始部署。几分钟后，你应该看到活动状态，如图 6-12 图 6-12 所示。

![Lightsail 部署成功](img/haad_0612.png)

###### 图 6-12\. Lightsail 部署成功

你的容器将有一个运行状态，并生成一个“公共域”（在图 6-12 中未显示）。这是 API 的基础 URL。为了验证你的 API 正在运行，从“公共域”中复制值并将其粘贴到浏览器栏中。你应该看到你的 API 的健康检查页面。恭喜你，你已经使用 AWS Lightsail 部署了你的 API！

# 更新你的 API 文档

在第五章中，你创建了*README.md*文件作为 API 的文档，但你还没有 API 地址。按照以下方式更新*README.md*文件的公共 API 地址部分，将`[*API URL*]`替换为你的部署应用的地址，无论你在哪里找到它：

```py
## Public API

Our API is hosted at [API URL]/([API URL]/). 

You can access the interactive documentation at [[API URL]/docs]([API URL]/docs).

You can view the OpenAPI Specification (OAS) file at
[[API URL]/openapi.json]([API URL]/openapi.json).
```

# 其他资源

[Docker 网站](https://oreil.ly/6AyzJ)提供了关于 Docker 容器概念的深入探讨。

对于 Python 容器化的最佳实践，请参阅 Snyk.io 的“使用 Docker 容器化 Python 应用程序的最佳实践”。

# 摘要

在本章中，你学习了使用云托管平台部署 API 的好处和责任。你现在有几种不同方式部署应用程序的经验：

+   在本地开发环境或 GitHub Codespaces 环境中

+   在本地开发环境或 GitHub Codespaces 环境中的 Docker 容器内

+   在 Render 云主机上

+   在 AWS 上使用 Amazon Lightsail 和 Docker

在第七章，你将创建一个 SDK，使 Python 开发者更容易使用你的 API。
