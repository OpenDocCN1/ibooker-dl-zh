
```py
from datetime import datetime
from itam.domain.asset import Asset
from itam.domain.hardware import Hardware
from itam.domain.funding_details import FundingDetails
from itam.domain.depreciation_strategy import (
    DepreciationStrategy,
    StraightLineDepreciationStrategy,
    DecliningBalanceDepreciationStrategy,
    DoubleDecliningDepreciationStrategy,
    NoDepreciationStrategy
)

class AssetFactory:
    def __init__(self):
        self.id = 0

    def date_from_string(self, date_string: str) -> datetime:
        return datetime.strptime(date_string, "%Y-%m-%d")

    def new(self, asset_type: str, asset_name: str, 
asset_cost: float, useful_life: int, 
depreciation_strategy: str, depreciation_rate: 
float, salvage_value: float, 
asset_purchase_date: str) -> Asset:
        self.id += 1
        purchase_date = self.date_from_string(asset_purchase_date)

        if asset_type == "hardware":
            return HardwareAssetFactory().new(self.id, 
asset_name, asset_cost, useful_life, 
depreciation_strategy, 
depreciation_rate, salvage_value, 
purchase_date)
        elif asset_type == "software":
            return SoftwareAssetFactory().new(self.id, 
asset_name, asset_cost, useful_life, 
depreciation_strategy, depreciation_rate, 
salvage_value, purchase_date)
        else:
            raise ValueError
```

`new` 方法根据 `asset_type` 参数确定需要调用哪个特定工厂。此外，它还跟踪已创建的资产数量，确保标识符（`id` 字段）始终唯一。否则，如果标识符的创建被推到特定的 `Factory` 对象，我们将会得到重复的键。

列表 4.8 内嵌的 `HardwareAssetFactory` 类

```py
class HardwareAssetFactory:
    def new(self, id: int, asset_name: 
str, asset_cost: float, 
useful_life: int, 
depreciation_strategy: str, 
depreciation_rate: float, 
salvage_value: float, 
purchase_date: datetime) -> Asset:
        a = Hardware (
            id=id,
            name=asset_name,
            category="hardware",
            cost=asset_cost,
            useful_life=useful_life,
            status="active",
            salvage_value=salvage_value,
            purchase_date=purchase_date
        )

        funding_details=FundingDetails(
                asset=a,
                depreciation_strategy=
                    self._get_depreciation_strategy(depreciation_strateg,
                depreciation_rate=depreciation_rate,
                department_allocations=dict()
            )

        a.funding_details = funding_details
        return a

    def _get_depreciation_strategy(self, 
depreciation_strategy: str) -> DepreciationStrategy:
        if depreciation_strategy == "straight_line":
            return StraightLineDepreciationStrategy()
        elif depreciation_strategy == "double_declining_balance":
            return DoubleDecliningDepreciationStrategy()
        elif depreciation_strategy == "none":
            return NoDepreciationStrategy()
        elif depreciation_strategy == "declining_balance":
            return DecliningBalanceDepreciationStrategy()
        else:
            raise ValueError
```

`HardwareAssetFactory` 类的 `new` 方法相对简单。这个方法接受来自 `AssetFactory` 的参数，尝试解析 `DepreciationStrategy`，并设置一些合理的默认值。

列表 4.9 内嵌的 `SoftwareAssetFactory` 类

```py
class SoftwareAssetFactory:
    def new(self, id: int, asset_name: str, 
asset_cost: float, 
useful_life: int, 
depreciation_strategy: str, 
depreciation_rate: float, 
salvage_value: float, 
purchase_date: datetime) -> Asset:
        a = Asset(
            id=id,
            name=asset_name,
            category="software",
            cost=asset_cost,
            useful_life=useful_life,
            status="active",
            salvage_value=salvage_value,
            purchase_date=purchase_date
        )

        funding_details=FundingDetails(
            asset=a,
            depreciation_strategy=self.
_get_depreciation_strategy(depreciation_strategy),
            depreciation_rate=depreciation_rate,
            department_allocations=dict()
        )

        a.funding_details = funding_details
        return a

    def _get_depreciation_strategy(self, 
depreciation_strategy: str) -> DepreciationStrategy:
        if depreciation_strategy == "straight_line":
            return StraightLineDepreciationStrategy()
        elif depreciation_strategy == "double_declining_balance":
            return DoubleDecliningDepreciationStrategy()
        elif depreciation_strategy == "none":
            return NoDepreciationStrategy()
        elif depreciation_strategy == "declining_balance":
            return DecliningBalanceDepreciationStrategy()
        else:
            raise ValueError
```

`SoftwareAssetFactory` 类几乎与 `HardwareAssetFactory` 类完全相同——如此相似，以至于可能存在代码异味。在软件开发中，*代码异味*这个术语描述的是开发者的一种直觉，表明代码可能存在更深层次的问题。这本身不是一个错误，而是一种感觉。你可能会有重构的冲动，因为这似乎违反了 DRY 原则（不要重复自己）。

但有一种更简单的方式来处理这种去重。为此，我们将查看下一个设计模式：*建造者模式*。

## 建造者模式

建造者模式是一种创建型设计模式，它通过提供如何创建对象的逐步说明，提供了一个流畅的 API 来创建对象。

### 4.2.3 指示系统如何构建

首先，我们将编写一个提示，让 Copilot 创建我们的构建器：一个用于 `Asset`，另一个用于 `FundingDetails`。我们将让构建器知道，如果 `asset_type` 是硬件，它应该返回一个 `Hardware` 的实例，对于 `Software` 也是如此：

```py
# Create a class called AssetBuilder
# It should use the Builder pattern to build an Asset
# Create another class called FundingDetailsBuilder
# It should use the Builder pattern to build a FundingDetails
# The AssetBuilder should have an embedded FundingDetailsBuilder
# When the category is "hardware" the AssetBuilder should create a Hardware object
# When the category is "software" the AssetBuilder should create a Software object
# When depreciation_strategy is equal to "straight_line" it should return a new instance of StraightLineDepreciationStrategy
# When depreciation_strategy is equal to "double_declining_balance" it should return a new instance of DoubleDecliningDepreciationStrategy
# When depreciation_strategy is equal to "none" it should return a new instance of NoDepreciationStrategy
# When depreciation_strategy is equal to "declining_balance" it should return a new instance of DecliningBalanceDepreciationStrategy
# The variables will need to be held in local variables and then passed to the Asset and FundingDetails objects when they are created.
# The final method of the AssetBuilder should return an Asset and be called build().
# The final method of the FundingDetailsBuilder should return a FundingDetails and be called build().
```

有一个需要注意的事项是，所有值都需要存储在局部变量中；否则，我们可能会遇到初始化后的异常。下面展示了 `AssetBuilder` 生成的代码。为了简洁起见，省略了 `import` 语句和访问器方法。值得一提的是，这些方法返回 `self`，因此可以串联起来使用。

列表 4.10 对应的 `AssetBuilder` 类

```py
class AssetBuilder:
    def __init__(self):
        self.id = 0
        self.name = ""
        self.category = ""
        self.cost = 0.0
        self.useful_life = 0
        self.status = ""
        self.salvage_value = 0.0
        self.purchase_date = datetime.now()

    def with_name(self, name):
        self.name = name
        return self

    ...
```

接下来，我们将检查 `AssetBuilder` 类的 `build()` 方法。这个方法使用 `category` 字段来返回正确的 `Asset` 子类。

列表 4.11 `AssetBuilder` 类的 `build()` 方法
