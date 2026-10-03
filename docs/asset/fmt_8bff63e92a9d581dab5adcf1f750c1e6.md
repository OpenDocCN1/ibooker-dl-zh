
我们定义了一些我们将与用于文件的模拟数据关联的批次。具体来说，当有人查询 API 端点，如`/api/v1/files/{fileId}/batches`时，我们希望返回这些数据。正如我们在 WireMock 中看到的，我们可以为所有我们感兴趣的`fileId`定义硬编码的路径。然而，这将导致大量的映射。相反，我们可以使用模板动态匹配这些字段，并为每个返回特定的响应。我们在以下列表中定义了`ach_files_batches.json`。请注意，我们使用`urlPathTemplate`并指定`{fileId}`。然后我们还在响应的`bodyFileName`中使用`{{request.path.fileId}}`，这允许我们提取我们期望接收的动态 UUID，并在`__files`目录中定义的响应中使用它。

#### 列表 6.30 WireMock 模板示例

```py
{
    "request": {
        "method": "GET",
        "urlPathTemplate": "/api/v1/files/{fileId}/batches"  #1
    },
    "response": {
        "status": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "bodyFileName": "ach_files_batches
➥{{request.path.fileId}}.json" #2
    }
}
```

#1 使用`urlPathTemplate`允许使用参数

#2 `request.path.fileId`引用在`urlPathTemplate`中定义的变量。

所有的这些，我们将能够定义对 API 调用的特定响应。图 6.6 更好地说明了这个概念。请注意，我们实际上并没有将任何东西保存到 WireMock 中。我们正在使用预先填充在 WireMock `__files` 目录中的预定义响应。

![一个图表的截图  自动生成的描述](img/CH06_F06_Kardell.png)

#### 图 6.6 WireMock 序列

现在我们能够返回数据，我们可以定义一个可以显示数据的组件。

### 6.6.2 更新仪表板以跳转到批次屏幕

我们更新了`RecentAchUploads.tsx`，使文件名成为一个链接。

#### 列表 6.31 更新的`RecentAchUploads`

```py
<TableCell>
   <Link onClick={() =>                      #1
      route.push(`/fileDetails/${row.id}`)}>  #1
         {row.filename}                      
   </Link>
</TableCell>
```

#1 将文本更新为使用文件 ID 的链接

`fileDetails`屏幕接受文件的 UUID，这将有助于驱动生成的查询。Next.js 使用基于文件的路由，允许我们使用简单的文件夹结构定义 URL。在这种情况下，将 UUID 作为 URL 的一部分表示为方括号中的目录。以下列表是我们需要创建的文件夹结构。

#### 列表 6.32 文件详情页面的文件夹结构

```py
src/
├─ app/
│  ├─ fileDetails/
│  │  ├─ [id]/          #1
│  │  │  ├─ page.tsx
```

#1 方括号表示我们可以访问的 URL 参数。

`[id]`是我们 URL 中的一个动态参数。例如，使用这种结构，我们应该能够导航到类似于`/fileDetails/0854b018-c5ce-4e00-8a4f-d268a1a85aa2`或`/fileDetails/d11fb49c-4a07-4b96-88cc-db1fc7277bfa`的 URL，假设这些是我们 ACH 文件的有效 UUID。

### 6.6.3 显示批次详情

`page.tsx`将包含检索批次信息并显示它的逻辑。

#### 列表 6.33 批次详情

```py
…
export interface AchBatchInfo {           #1
    companyName: string;                   #2
    companyBatches: AchBatchDetails[];     #2
}
…
export interface AchBatchDetails {    
    batchId: string;                  
    companyName: string;              
    recordCount: number;              
    totalDebitAmount: number;         
    totalCreditAmount: number;        
}
…
interface AchBatchInfoResponse {  
    batches: AchBatchInfo[];      
}
…
export default function FileDetails({params}: any) { #4

    const id: string | string[] = params.id; #5
    const [batches, setBatches] =
➥ useState<AchBatchInfo[]>([]); #6

    useEffect(() => {        #7
        const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? '';
        axios.get<AchBatchInfoResponse>(`${apiUrl}/files/${id}/batches`)
            .then(response => {
                console.log(`Response data
➥ ${JSON.stringify(response.data.batches)}`);
                setBatches(response.data.batches);
            })
            .catch(error => {
                console.log(error);
            });
    }, []);
…
                    {batches.map((item, index) => (          #8
                            <Accordion key={item.companyName}>
                                <AccordionSummary
➥ expandIcon={<ExpandMoreIcon />}>
                                    <Typography>{item.companyName}</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
{item.companyBatches.map((companyBatch, index) => (
                                    <>
                                        <Typography>
                                  Batch ID:
➥ {companyBatch.batchId}<br />
                                  Record Count:
➥ {companyBatch.recordCount}<br />
                                  Total Debit Amount:
➥ {formatCurrency(companyBatch.totalDebitAmount)}<br />
                                  Total Credit Amount:
➥ {formatCurrency(companyBatch.totalCreditAmount)}
                                        </Typography>
                                        <Link>View Batch</Link>
                                        <Divider />
                                    </>
                                )
                                )}
                                </AccordionDetails>
                            </Accordion>
                        ))}
…
```

#1 定义我们的 JSON 结构，以便我们可以对响应进行强类型化

#4 允许访问 URL 参数

#5 通过与我们在方括号中定义的相同名称引用 URL 参数

#6 使用`useState`类似我们在其他组件中做的那样存储批次

#7 使用`useEffect`从 API 获取数据

#8 将返回的批次映射到显示 UI 元素

此组件展示了本章前面提到的作为 6.5 图的一部分的折叠视图，用于 ACH 文件中的公司。从这个组件中我们可以得到的的一个重要启示是看到我们如何可以在页面上动态创建布局元素。在其他组件中，我们将使用提供表格结构以查看动态数据的 DataGrid 视图。然而，了解我们可以在需要时手动创建具有动态数据的布局也是有帮助的。

## 6.7 创建散点图组件

最后，我们想在页面上添加一个散点图，显示批次中的项目与借方和贷方总额的关系。以下列表显示了我们将要创建的目录结构。您可能需要创建另一个子目录来存储特定的散点图相关文件。由于我们尚未达到这个目录包含大量文件的程度，我们将暂时保持原样。

#### 列表 6.34 我们图表的目录结构

```py
src/
├─ app/
│  ├─ components/
│  │  ├─ charts/
│  │  │  ├─ AchBatchesScatterPlot.tsx
│  │  │  ├─ AchBatchesScatterTooltip.tsx
│  │  │  ├─ CurrencyTick.tsx
```

### 6.7.1 散点图组件

散点图有助于展示变量之间可能的相关性。在我们的案例中，我们可能并不一定对分析方面感兴趣。然而，散点图也提供了一种识别具有大量项目或大量金额的批次的途径，这可能会对我们的用户很有趣。当然，我们也许还有机会展示特定公司的批次趋势，这也许有助于我们的用户进行数据分析。

同时，这个组件提供了探索使用颜色、自定义提示和自定义轴刻度的机会。这些特性将在后续章节中讨论。散点图的初始代码显示在列表 6.35 中，实际组件渲染显示在列表 6.36 中。

组件定义了一个名为`AchBatchesScatterPlotProps`的接口，以便它可以正确设置类型。此外，这个组件接受批处理作为只读参数，而不是在组件内部进行任何 API 调用。我们本可以在组件内部进行 API 调用。然而，如果我们有多个组件想要使用这些数据，我们最终会进行重复的 API 调用来收集相同的数据。我们还标记数据为只读，以帮助强制执行组件仅利用数据的概念，并通过使用不引入意外后果。然而，请注意`Readonly`是 TypeScript 的编译时特性，生成的 JavaScript 不会有这个特性。最后，由于我们感兴趣的是批次的总金额，我们将信用和借记金额相加，并将它们存储到`totalCreditAndDebits`中。

#### 列表 6.35 散点图组件

```py
…
interface AchBatchesScatterPlotProps {  #1
    batches: AchBatchInfo[];             #1
}                                       

export default function AchBatchesScatterPlot({batches}:
➥ Readonly<AchBatchesScatterPlotProps>) { #2

    const plotData = batches.map(batch => {
        const updatedCompanyBatches =
➥ batch.companyBatches.map(companyBatch => {
            return {
                ...companyBatch,   #3
                totalCreditAndDebits:
➥ companyBatch.totalCreditAmount + companyBatch.totalDebitAmount #4
            };
        });
        return {
            ...batch, #5
            companyBatches: updatedCompanyBatches #6
        };
    });

    return (
    # Refer to AchBatchesScatterPlot.tsx source code for this logic
    );
}
```

#1 定义组件属性的接口，以便我们可以进行强类型化

#2 作为只读属性接受属性，所以我们不会修改它们

#3 将`companyBatch`对象的内容展开

#4 向对象添加一个名为`totalCreditAndDebits`的新属性

#5 将批处理对象的内容展开

#6 用包含`totalCreditAndDebits`的新批处理替换现有的`companyBatches`

在下面的列表中，我们有渲染实际散点图的组件。我们利用 recharts 的灵活性，通过传递一个名为`AchBatchesScatterTooltip`的组件来定义自定义提示内容，并定义一个用于`YAxis`的自定义刻度标记组件，称为`CurrencyTick`。此外，我们在图上放置点时为每个公司分配不同的颜色（关于这些自定义组件的更多内容将在后续章节中介绍）。

#### 列表 6.36 散点图渲染

```py
<>
   <Title>ACH Batches</Title>
   <ResponsiveContainer>
      <ScatterChart
         width={400}
         height={400}
         margin={{
            top: 20,
            right: 20,
            bottom: 20,
            left: 20,
         }}>
      <Legend />
      <CartesianGrid />
      <XAxis type="number" dataKey="recordCount" name="Count" unit="" />
      <YAxis type="number"
             tick={<CurrencyTick/>}   #1
             dataKey="totalCreditAndDebits" name="Total Amount" unit="" />
      <Tooltip cursor={{ strokeDasharray: '3 3' }} 
         content={<AchBatchesScatterTooltip />} />   #2
         {plotData.map((entry, index) => (
            <Scatter key={`${entry.companyName}`} 
                     name={`${entry.companyName}`} 
                     data={entry.companyBatches} 
                     fill={stringToColor(entry.companyName)}>  #3
               <Cell 
                  key={`cell-${entry.companyName}-${index}`} 
                            fill={stringToColor(entry.companyName)} />  
                      </Scatter>
                   ))}
      </ScatterChart>
   </ResponsiveContainer>
</>
```

#1 用于 Y 轴刻度的自定义组件

#2 当悬停在数据点上时自定义提示

#3 根据公司自定义颜色

### 6.7.2 散点图自定义提示
