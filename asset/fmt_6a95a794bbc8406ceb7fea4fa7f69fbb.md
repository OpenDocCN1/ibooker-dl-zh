
# 第十章：模型部署

本章涵盖

+ 在我们的本地系统上部署深度学习模型到简单的网络应用程序
+ Google Cloud 关键概念的介绍
+ Vertex AI 简介，Google Cloud 中的机器学习环境
+ 使用 Vertex AI 端点部署深度学习模型
+ 适配网络应用程序以使用 Vertex AI 端点
+ 使用 Gemini 为 Google Cloud 获取生成式 AI 辅助

在第九章中，我们回顾了使用表格数据训练深度学习模型的一组最佳实践，并介绍了吉隆坡房地产价格预测问题作为一个具有混合类型特征的挑战性表格问题。在本章中，我们将使用第九章中训练的模型，并在一个简单的网络应用程序中部署它。首先，我们将本地部署它——也就是说，将网络服务器和训练好的模型都放在我们的本地系统上。接下来，我们将介绍 Google Cloud 作为部署我们模型的另一种方式。实际上，我们将训练好的模型部署到 Vertex AI 中，这是 Google Cloud 中的机器学习环境。最后，我们将探讨如何在 Google Cloud 上使用 Google 的生成式 AI 助手 Gemini。本章中描述的代码可在 [`mng.bz/6e1A`](https://mng.bz/6e1A) 获取。

## 10.1 简单的 Web 部署

一旦我们训练了一个深度学习模型，并遵循了第九章中描述的最佳实践，我们仅仅只是触及了从模型中获取价值的过程的表面。以我们在第九章中训练的吉隆坡房地产价格预测模型为例。这个模型对于想要为他们的客户提供定价建议的房地产经纪人来说可能很有用。这个模型也可能对想要将他们的房产推向市场以了解他们可能期望从其房产中获得的价值的房主有用。最后，这个模型也可能对对在吉隆坡购买房产感兴趣的买家有用，这样他们就可以根据一定的预算了解在特定位置可以购买哪种类型的房产。

显然，我们在第九章训练的模型有可能对各种受众都有用。我们面临的困境是如何让这个模型对所有具有不同商业目标和可能不同计算机技能的受众都可用。在本章的后面部分，我们将学习如何在公共云环境中部署模型，但如果我们想进行一个初步实验来了解模型的特点并与一些测试客户进行测试呢？我们需要实现整个在公共云中部署模型的过程吗？不，因为我们可以利用 Flask，这是一个 Python 网络应用程序库，并设置一个自包含的模型网络部署。Flask 最初由 Armin Ronacher 于 2010 年发布，并从此成为 Python 最受欢迎的 Web 框架之一。考虑到最近出现的包，如 Streamlit ([`streamlit.io/`](https://streamlit.io/)) 或 Gradio ([`www.gradio.app/`](https://www.gradio.app/))，它可能听起来有点过时。然而，尽管它已经存在了一段时间，但由于其轻量级和灵活的特性，Flask 仍然是 Web 开发的合适选择，尤其是对于小型个人项目。我们不会在本书中探讨 Flask 的所有细节；我们只是提供设置模型演示所需的要点，但如果你想了解更多关于 Flask 的信息，请查看文档：[`flask.palletsprojects.com/en/2.3.x/`](https://flask.palletsprojects.com/en/2.3.x/)。

现在我们已经介绍了简单 Web 部署的概念，我们可以开始着手了。在本节中，我们将介绍如何使用 Flask 在两个网页中创建一个简单而完整的模型部署。

### 10.1.1 网络部署概述

到目前为止，在这本书中，我们已经探讨了各种机器学习和深度学习的库和框架。除了在 YAML 中定义的配置文件外，我们在这本书中查看的所有代码都是 Python 编写的。像我们之前看到的例子一样，机器学习或深度学习模型的 Web 部署确实包括 Python，但它还需要处理 HTML、JavaScript 和 CSS 脚本。图 10.1 展示了我们心中的网络部署概述。

![](img/CH10_F01_Ryan2.png)

图 10.1 网络部署概述

让我们回顾一下构成图 10.1 所示网络部署的组件：

+ *训练好的模型*—这是我们保存在 Keras 预处理层笔记本中的模型。实际上，模型是以目录结构保存的 ([`mng.bz/oKJp`](https://mng.bz/oKJp))。在 Flask 服务器模块中，使用对该目录的引用来加载模型。
+ *Flask 服务器模块*—这是一个 Python 模块，用于加载训练好的模型，并包含每个解决方案 HTML 页面的视图函数。视图函数指定当这些 HTML 页面被加载时，Flask 服务器模块采取的操作。
+ *Flask 服务器配置文件*—YAML 文件，其中可以指定 Flask 服务器的参数，例如包含训练模型的目录。
+ `home.html`—由 Flask 服务器提供的 HTML 页面之一。此页面包含用户可以指定他们想要获取价格预测的房产特征的字段（例如位置、房间数量和停车位数量）。此页面还包含 JavaScript 函数，用于将默认值加载到页面上的每个字段中，指定可以输入到每个字段的合法值，并将用户输入的值打包，以便可以发送回 Flask 服务器模块。
+ `show-prediction.html`—由 Flask 服务器提供的 HTML 页面之一。此页面显示模型对用户在`home.html`页面上输入的房产特征的预测。
+ *CSS 文件*—指定解决方案中 HTML 页面的渲染方式。

与本书中的其他示例不同，默认情况下 Web 部署运行在您的本地系统上。当 Flask 服务器运行时，它会在`localhost:5000`上提供`home.html`，这样您就可以在浏览器中练习系统。

现在我们已经介绍了 Web 部署，我们将在后续小节中更详细地查看 Flask 服务器模块和 HTML 页面。

### 10.1.2 Flask 服务器模块

Flask 服务器模块是 Web 部署的核心。与本书中迄今为止我们检查的其他 Python 程序不同，它是一个独立的`.py`文件，而不是 Jupyter Notebook。

在本小节中，我们将回顾 Flask 服务器模块中的关键代码片段，并解释它们如何驱动 Web 部署。以下列表显示了 Flask 服务器模块的第一个关键部分：从文件系统将保存的 Keras 模型加载到 Python 模块中的对象的代码。

列表 10.1 加载保存的 Keras 模型

```py
rawpath = os.getcwd()                                             ①
model_path = os.path.abspath(os.path.join(rawpath, 'models'))
model_directory_name = \
os.path.join(model_path,config['file_names'] \
['saved_model'])                                                  ②
loaded_model = \ 
tf.keras.models.load_model(model_directory_name)                  ③
```

① 获取当前目录

② 获取完全限定的模型目录

③ 使用完全限定的模型文件名加载模型

在列表 10.1 中，完全限定的模型目录是通过从配置文件中加载的模型文件名构建的。期望模型目录位于名为`models`的目录中，与 Flask 服务器模块在同一目录下。模型是通过与第九章中 Keras 预处理层训练笔记本中用于加载模型的相同`tf.keras.models.load_model`函数加载的。

Flask 服务器模块最重要的部分是视图函数，它们指定了应用程序中每个 HTML 页面采取的操作。以下列表显示了`home.html`的视图函数，这是当`home.html`是目标时调用的代码。

列表 10.2 `home.html`视图函数

```py
@app.route('/')                                                 ①
def home():   
    title_text = "web deployment of Keras model"
    title = {'titlename':title_text}
    return render_template('home.html',title=title)             ②
```

① 装饰器，表示此视图函数用于`home.html`

② 发送到`home.html`以进行渲染的参数

列表 10.2 中显示的视图函数只是向`home.html`页面发送一个标题，并渲染页面。

列表 10.3 显示了`show-prediction.html`的视图函数。当`show_prediction.html`是目标时，Flask 模块运行此代码。此视图函数处理用户在`home.html`中输入的值，并调用模型对这些值进行预测。

列表 10.3 `show-prediction.html`视图函数

```py
@app.route('/show-prediction/')                                 ①
def show_prediction():
    scoring_dict = {}
    for col in config['scoring_columns_cat']:
        print("value for "+col+" is: "+str(request.args.get(col))) 
        scoring_dict[col] = str(request.args.get(col))          ②
    for col in config['scoring_columns_cont']:
        scoring_dict[col] = float(request.args.get(col))        ③
    scoring_dict['size_type_bin'] = \
str(request.args.get('size_type'))+' 1'                         ④
    input_dict = \
{name: tf.convert_to_tensor([value]) for name, \
value in scoring_dict.items()}                                  ⑤
    predictions = loaded_model.predict(input_dict)              ⑥
    prob = tf.nn.sigmoid(predictions[0])
    if prob < 0.5:                                              ⑦
        predict_string = "Prediction is: property has a price less than 
median"
    else:
        predict_string = "Prediction is: property has a price more than 
median"
    prediction = {'prediction_key':predict_string}
    return(render_template('show-prediction.html',\
prediction=prediction))                                         ⑧
```

① 装饰器表示此视图函数用于`show-prediction.html`

② 将查询字符串中的分类特征值加载到`scoring_dict`字典中

③ 将连续特征值从查询字符串加载到`scoring_dict`字典中

④ 设置`size_type_bin`参数的值

⑤ 从`scoring_dict`构建输入字典

⑥ 从加载的模型中获取加载的特征值的预测

⑦ 根据预测值设置输出预测字符串

⑧ 使用预测字符串渲染`show-prediction.html`

列表 10.3 中显示的视图函数将`home.html`返回的查询字符串加载到 Python 字典`scoring_dict`中，并使用这些值调用列表 10.1 中代码加载的模型以获取预测。唯一的例外是`size_type_bin`。为了这个简单的 Web 部署，我们硬编码了分箱值。作为一个练习，考虑一下你将如何计算分箱值以避免这种硬编码。预测的值用于设置一个字符串值，该值在页面渲染时发送到`show-prediction.html`以显示。

图 10.2 说明了列表 10.2 和 10.3 中显示的视图函数如何与`home.html`和`show-prediction.html`网页交互。

![](img/CH10_F02_Ryan2.png)

图 10.2 视图函数与网页之间的交互

图 10.2 解释了 Flask 服务器模块中`home.html`的视图函数渲染`home.html`。`home.html`中的 JavaScript 函数构建一个包含用户在`home.html`中的特征值的查询字符串。这个查询字符串返回到 Flask 服务器模块中的`show-prediction`视图函数。该视图函数从查询字符串中加载特征值，并使用它们从模型中获取属性预测。模型预测用于创建预测字符串，当渲染时传递给`show-prediction.html`。预测字符串在`show-prediction.html`中显示。

现在我们已经检查了图 10.2 右侧 Flask 服务器模块中的 Python 代码的关键部分，在下一节中，我们将超越 Python，查看驱动图 10.2 左侧发生的 HTML 和 JavaScript 代码。

### 10.1.3 `home.html`页面

`home.html`页面中发生了很多事情：

+ 设置用户可以输入基于预测的八个属性特征的值的字段
+ 为每个字段设置默认值和范围
+ 当用户点击获取预测并从字段收集输入时，将它们发送到 Flask 服务器模块中的视图函数

图 10.3 总结了 `home.html` 中 JavaScript 函数与用户界面元素的交互。

![](img/CH10_F03_Ryan2.png)

图 10.3 `home.html` 中 JavaScript 和 UI 元素的交互

在介绍了 `home.html` 中的情况之后，让我们来检查这个网页的代码。列表 10.4 展示了 `home.html` 中输入字段定义的示例。这些是用户可以输入值以指定他们想要获取价格预测的属性的详细信息的 HTML 字段。

列表 10.4 字段 HTML 定义示例

```py
<p> 
    Location:                                              ①
    <select id="location" >                                ②
    </select> 
</p>
<p> 
    <label for="rooms">
Number of rooms (1 - 20):</label>                          ③
        <input type="number" id="rooms" name="rooms"       ④
        min="1" max="20">                                  ⑤
</p>
```

① 位置字段的标签

② 定义位置字段的 ID

③ 定义房间字段的标签

④ 定义房间字段的 ID

⑤ 定义房间字段的最大值和最小值

列表 10.4 展示了用于分类字段（`location`）和连续字段（`rooms`）的 HTML 定义示例。其他字段的 HTML 定义遵循相同的模式，为所有连续字段设置了最小值和最大值。

当页面通过以下语句在 HTML 页面开头加载时，`load-selections()` JavaScript 函数会被执行：

```py
<body onload="load_selections()">
```

以下列表展示了 `load-selections()` JavaScript 函数的关键部分，该函数设置页面以便用户可以选择他们想要获取预测的属性的属性。

列表 10.5 `load-selections()` JavaScript 函数
