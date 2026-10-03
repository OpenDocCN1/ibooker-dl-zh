
```py
export interface AchTransactionSearchResponse { #1
    file_id: string;  #2
    batch_header_id: string;  #2
    entry_id: string;  #2
    filename: string;  #2
    individual_name: string;  #2
    amount: string;  #2
}
```

#1 导出`AchTransactionSearchResponse`使其在所有地方可用

#2 API 返回的字段

一旦我们有一个地方来存储检索到的数据，我们就可以创建一个组件来显示这些数据。这个`AchTransactionSearchResults`组件将类似于我们创建的其他组件，我们将通过`AchTransactionSearchResultsProps`的结果对象将 API 调用的结果传递给组件。我们利用列定义上的`renderCell`属性添加链接，允许用户直接跳转到特定搜索结果的文件或批次。此外，由于我们的结果没有 ID 列，我们必须使用`getRowId`属性在 DataGrid 组件中定义一个。`entry_id`可以作为我们的 ID 列，因为对于每个结果，`entry_id`是交易的 UUID，它是唯一的。以下列表显示了组件的更重要的部分。

##### 列表 9.15  `AchTransactionSearchResults` 组件

```py
interface AchTransactionSearchResultsProps {
    results: AchTransactionSearchResponse[];
}

export default function AchTransactionSearchResults #1
➥({results}:  #1
Readonly<AchTransactionSearchResultsProps>) {  #1

    const route = useRouter(); #2

    const columns: GridColDef[] = [
        {field: 'filename', headerName: 'Filename', #3
➥ width: 150}, 
        …
        {field: 'viewFile', headerName: '',  #4
                 sortable: false, width: 150,   #4
                 renderCell: (params) => (   #4
            <Link onClick={() =>  #4
                route.push(`/fileDetails/${params.row.file_id}`)}  #4
                sx={{ cursor: 'pointer' }}>Jump to file...</Link>  #4
            )},  #4
        {field: 'viewBatch', headerName: '', #5#6
                 sortable: false, width: 150,   #6
                 renderCell: (params) => (  #6
            <Link onClick={() =>   #6
route.push(`/fileDetails/${params.row.file_id}/   #6
➥batchDetails/${params.row.batch_header_id}`)}   #6
              sx={{ cursor: 'pointer' }}>Jump to batch...</Link>  #6
            )}  #6
    ];

    return (
…
       <DataGrid columns={columns} 
                 rows={results} 
                 getRowId={(row) => row.entry_id} #7
       />
…
    );
}
```

#1 导出搜索结果函数并声明参数为只读

#2 使用路由进行导航目的

#3 开始定义要显示的列

#4 定义一个列，该列将创建一个链接，点击后将导航到文件详情页面

#5 定义一个列，该列将创建一个链接，点击后将导航到批量详情页面

#6

#7 由于没有名为 id 的具体参数，我们指定哪个字段可以使用。

现在我们有了存储数据的地方和展示数据的方式，剩下要做的就是调用 API 并将其传递给我们的组件。为此，我们更新了我们的搜索页面，添加了一个文本输入字段和一个搜索按钮来执行实际的 API 调用。我们使用 `TextField` 的 `onChange` 事件将用户输入的文本存储到 `searchCriteria` 字段中。当用户点击搜索按钮时，将触发 `onClick` 事件，并执行 `handleSearch`。`handleSearch` 调用我们的 API 并存储结果。更新的搜索页面的重要部分如下所示。

##### 列表 9.16  更新的搜索页面

```py
…
export default function SearchPage() {
    const [searchCriteria, setSearchCriteria] = #1
➥ useState<string>(''); 
    const [results, setResults] = #2
useState<AchTransactionSearchResponse[]>([]); 

    const handleChange = (event: { target: { value: #3
React.SetStateAction<string>; }; }) => {  #3
        setSearchCriteria(event.target.value);  #3
    };  #3

    const handleSearch = async () => { #4
        …        axios.get<AchTransactionSearchResponse[]> #5
➥(`${apiUrl}/files/transactions/search?criteria=${searchCriteria}`, { 
…
    return (
…
       <Box sx={{display: 'flex', 
                 flexDirection: 'column', 
                 alignItems: 'center', gap: 2}}>
          <TextField id="search" #6
                     label="Search"  #6
                     variant="standard"   #6
                     onChange={handleChange}   #6
                     sx={{ width: '40%' }} />  #6
          <Button variant="outlined" #7
                  color="primary" onClick={handleSearch}>  #7
             Search #7
          </Button> 
       </Box>
       <AchTransactionSearchResults results={results}/> #8
…
    );
}
```

#1 将搜索标准初始化为空字符串

#2 将结果初始化为空字符串

#3 `handleChange` 函数将输入的文本保存以供以后使用。

#4 `handleSearch` 函数调用 API 并使用 `setResults` 保存结果。

#5 使用 `axios` 获取响应

#6 定义一个用于用户输入搜索字符串的 `TextField` 元素

#7 定义一个将搜索输入字符串的按钮元素

#8 一个用于显示结果的组件

现在我们有一个可以执行基本交易搜索并提供直接跳转到找到交易的文件或批次的搜索页面。接下来，我们通过 Playwright 加强测试，以获取我们搜索功能的一些集成级别测试。

### 9.2.4 使用 Playwright 进行集成测试

在我们使搜索我们的交易成为可能之后，添加一个集成测试将是一个很好的选择，我们可以用它来进行集成和回归测试。我们设置了一个测试，需要我们

1.  上传文件以解析

1.  导航到搜索页面

1.  填充搜索页面

1.  等待 ACH 响应

1.  验证结果是否已填充

1.  拍摄屏幕截图

我们创建了一个名为 `test_search_page.py` 的 Python 脚本来执行所有这些必要的步骤。此外，我们希望有选项可以在浏览器窗口显示的情况下调试我们的测试；为了实现这一点，我们需要创建一些 Pytest 固定装置。

##### 列表 9.17  Playwright 的 Pytest 固定装置

```py
@pytest.fixture(scope="session") #1
def browser(): #2
    with sync_playwright() as p:  #3
        browser = p.chromium.launch(headless=False) 
        yield browser #3
        browser.close() #4

#5
@pytest.fixture(scope="function") 
def page(browser): #6
  context = browser.new_context()  #7
 pages = context.pages
 page = pages[0] if pages else context.new_page() #7
  yield page #8
    context.close() #9
```

#1 定义一个具有会话范围的固定装置，这将导致固定装置在测试会话期间持续存在

#2 定义一个名为 `browser` 的函数，用于创建浏览器实例。我们使用 `headless=False` 以逐步进行测试会话并查看浏览器正在做什么。使用 `headless=True` 将隐藏浏览器窗口，这是运行单元测试的默认方式。

#3 `yield` 浏览器命令将返回浏览器对象。

#4 当所有测试完成后，关闭浏览器。

#5 定义一个具有函数范围的固定装置，这是默认范围

#6 创建一个用于测试的页面

#7 如果存在，则使用默认选项卡；否则，创建一个新的选项卡

#8 使用 `yield` 返回页面，以便在测试中可用

#9 测试运行后清除上下文

这就足够设置 Playwright 结构了。现在，我们想要确保我们的数据库为空，并在开始集成测试之前加载我们的 ACH 文件。

##### 列表 9.18  清空数据库并加载我们的文件

```py
@pytest.fixture(autouse=True) #1
def setup_teardown_method():
    SqlUtils.truncate_all() #2

    ach_file = "./data/sally_saver.ach" #3
    file_path = get_absolute_path("./data/sally_saver.ach")  #3
    parser = AchFileProcessor()  #3
    ach_files_id = SqlUtils.create_ach_file_record #3
➥(ach_file, str(randint(1, 99999999)))  #3
    parser.parse(ach_files_id, file_path)  #3
    yield #3

def get_absolute_path(relative_path): #4
    return current_file_dir / relative_path
```

#1 定义一个设置为 `autouse` 为 `True` 的 fixture。这会导致它默认包含在每个测试中。

#2 清空我们的数据库

#3 将测试文件加载到数据库中

#4 确保路径正确，基于我们从哪里运行

现在设置完成，我们可以开始使用 Playwright 进行实际测试。

##### 列表 9.19  Playwright 测试

```py
def test_dashboard(page: Page):
    page.goto("http://localhost:3000/search") #1
    page.expect_navigation(wait_until="load") 
    expect(page).to_have_url("http://localhost:3000/search") #2
    search_criteria = page.get_by_role("textbox") #3
    search_criteria.fill("sally saver") 
    search_button = page.locator("#searchbtn") #4
    with page.expect_response("**/files/transactions/search*") #5
➥ as response_info:  #6
        search_button.click() 

    response = response_info.value #6
    assert response.status == 200
    search_result = page.locator('div[title="Sally Saver"]') #7
    expect(search_result).to_be_visible() 
    page.screenshot(path="screenshots/search_results.png") #8
```

#1 导航到搜索屏幕并确保页面已加载

#2 验证我们是否在正确的页面上

#3 找到文本框并输入文本“sally saver”

#4 找到搜索按钮

#5 预期当点击搜索按钮时会有 API 响应

#6 检查响应状态码为 200

#7 我们应该能够看到一个包含文本“Sally Saver”的条目。

#8 为永存性拍摄屏幕截图

现在你已经知道了如何执行一些集成测试，你可能需要调用 API 并验证结果。在下一节中，我们将探讨许多行业中开发的一个关键方面：保持审计跟踪。

## 9.3 审计用户交互

互联网安全中心提出的几个关键安全控制之一是审计日志管理（[`mng.bz/gaGE`](https://mng.bz/gaGE)）。虽然我们不会涵盖所有方面，但我们将至少从确保我们收集有关应用程序使用情况的数据开始。记住，收集数据是第一步。如果我们没有监控、审查和必要时接收警报的策略，日志就毫无用处。

许多时候，我们解决过的问题最初是因为有人识别到日志中发生错误而被引入了错误的方向。进一步的研究表明，错误已经发生数月（远至日志容易获取的时候）并且不是当前问题的源头。有时，在生产的紧急事件中，由于需要快速解决问题，你很容易陷入这种陷阱。通过适当的审计日志管理，你可以更好地理解基线应用程序行为，并监控应用程序滥用，以及其他许多好处。

尽管有许多现成的商业工具（COTS）如 Splunk、Dynatrace、DataDog 和 Sentry 可以帮助进行日志记录和可观察性，但我们主要关注将数据记录到数据库中，并在必要时从那里扩展。

### 9.3.1 数据库设计

如果我们想要开始进行审计，我们需要创建一个新的数据库表来存储我们的数据。这将基于您的应用程序需求以及您试图实现的目标。在我们的案例中，我们将跟踪进入系统的 API 请求。我们引入了一种新的数据类型来处理主机和网络——`ip_address`字段的`INET`数据类型。虽然使用字符串数据类型对于`ip_address`来说当然也可以工作，但 Postgres 提供了处理`INET`的额外比较和函数，这可以使生活更加轻松。例如，我们可以通过使用`WHERE ip_address` `<<` `'192.168.1.0/24'`来搜索给定范围内的所有地址。如果我们想查看来自我们怀疑滥用我们应用程序的地址列表的活动，这种搜索可能会有所帮助。以下列表显示了表的创建。

##### 列表 9.20  `audit_log`表

```py
CREATE TABLE audit_log ( #1
    audit_log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(), #2
    user_id VARCHAR(25) DEFAULT NULL,
    ip_address INET DEFAULT NULL, #3
    user_agent VARCHAR(255) DEFAULT NULL,
 http_request VARCHAR(10) DEFAULT NULL,
  url VARCHAR(255) DEFAULT NULL,
    message TEXT NOT NULL
);
```

#1 创建名为 `audit_log` 的表

#2 当记录使用 `TIMESTAMP` 数据类型创建时保存。我们还使用 `NOT NULL`，以确保字段必须始终被填充。最后，我们默认使用 Postgres 中的 `NOW()`函数，该函数将在记录插入时填充字段，如果没有提供值。

#3 创建一个名为 `ip_address` 的字段，其数据类型为 `INET`，这是一个方便处理网络地址的数据类型
