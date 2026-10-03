
```py
from itam.domain.asset import Asset
from itam.infrastructure.ports.asset_rest_port import AssetRestPort
from itam.service.asset_manager import AssetManager

class AssetRestAdapter(AssetRestPort):
    def __init__(self, asset_manager: AssetManager):
        self._asset_manager = asset_manager

    def read(self, asset_id: int) -> Asset:
        return self._asset_manager.read(asset_id)

    def create(self, asset: Asset) -> None:
        self._asset_manager.create(asset)

    def update(self, asset: Asset) -> None:
        self._asset_manager.update(asset)

    def delete(self, asset_id: int) -> None:
        self._asset_manager.delete(asset_id)

    def get_assets(self):
        return self._asset_manager.get_assets()
```

剩下的工作只是更新`AssetController`以移除其直接调用`AssetManager`方法的操作，并让`AssetController`调用适配器的操作，适配器再调用`AssetManager`的操作。从端口和适配器模式的关键要点来看，它抽象了系统驱动部分（在这种情况下，REST API）与驱动应用（我们的业务模型和系统核心`AssetManager`）之间的交互。为了使这一点更加明确，我们将很快再次修改这个类，将其端口添加到构造函数中。

列表 4.25 使用`AssetRestAdapter`更新`AssetController`的代码

```py
router = APIRouter()

asset_rest_adapter = AssetRestAdapter(AssetManager())
asset_factory = AssetFactory()

@router.post('/assets', response_model=AssetOut)
async def create_asset(asset_in: AssetIn):
    asset = asset_factory.new(asset_in.asset_type,
    asset_in.name, asset_in.unit_cost, 
    asset_in.useful_life, asset_in.depreciation_strategy,
    asset_in.depreciation_rate, asset_in.salvage_value,
    asset_in.purchase_date)
    asset_rest_adapter.create(asset)
    return from_asset(asset)
```

如前所述，我们将修改`AssetController`以移除所有对`AssetManager`的直接引用。我们的当前`AssetController`，尽管它没有直接调用任何`AssetManager`方法，但确实持有对`AssetManager`的间接引用，因为`AssetManager`是在`AssetController`中构建的。这样做进一步隔离了`AssetManager`与驱动程序的变化。

### 4.3.3 访问我们的数据和持久化我们的更改

软件中的抽象不仅仅是学术问题。这些抽象在满足需求和进行更改时给我们提供了灵活性。它们可以隐藏复杂性，将行为封装在看似简单的接口之后。这使我们来到了下一个模式：**仓库**模式。

仓库模式

仓库模式是一种设计模式，通过创建一个单独的地方来处理获取、保存、更新和删除数据等任务来帮助管理数据。它将数据源（如数据库或文件）连接到应用程序的其余部分，使数据访问代码与其他代码分离。

在我们的 ITAM 项目中，仓库模式将使我们能够提供一个大大简化的接口，掩盖了数据访问的复杂性。从仓库或`AssetManager`的角度来看，我们的数据访问层只是一系列方法。幕后，它可能是一个 REST API、Redis 缓存、MongoDB、文件或任何类型的数据库。让我们创建一个提示，使用仓库模式设置我们的通用数据访问层：

```py
# Create an abstract base class called BaseRepository that represents a generic data access layer using the Repository pattern. 
# The class should define common methods for CRUD operations such as create, read, update, and delete. 
# Make sure to use type hints for the input parameters and return values.
```

使用这个提示，Copilot 应该拥有建立所有实现都将遵守的合同所需的一切。让我们检查一下代码。

列表 4.26 `BaseRepository`的代码，我们的通用数据访问层

```py
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

T = TypeVar('T')

class BaseRepository(ABC, Generic[T]):
    @abstractmethod
    def create(self, entity: T) -> None:
        pass

    @abstractmethod
    def read(self, entity_id: int) -> T:
        pass

    @abstractmethod
    def update(self, entity: T) -> None:
        pass

    @abstractmethod
    def delete(self, entity_id: int) -> None:
        pass

    @abstractmethod
    def get_entities(self) -> List[T]:
        pass
```

Copilot 引入了一个`Generic`类型，允许在实现时提供一些类型提示。如果我们表达类型应该是`Asset`，方法签名将展示这一点。我们的第一次实现尝试将是一个内存数据存储。这将使我们能够对整个系统进行端到端测试。这也很好地展示了抽象的有用性。不久，我们将添加另一个使用 Postgres 的实现。但首先，这是提示：

```py
# Define a class called InMemoryAssetRepository that inherits from the BaseRepository abstract class. 
# The InMemoryAssetRepository should be specific to managing Asset objects using an in-memory list to store and update the assets. 
# Implement the necessary methods for CRUD operations such as create, read, update, and delete for Asset objects using the in-memory list. 
# Make sure to use type hints for the input parameters and return values.
```

我们的内存实现将使用一个简单的列表来存储`Asset`。我们将能够执行所有 CRUD 操作。此外，Copilot 将为我们添加类型提示，让其他人知道它应该接受和返回`Asset`。这就是`BaseRepository[Asset]`的意思：它是一个`BaseRepository`的实现，其`Generic`类型是`Asset`。

列表 4.27 `InMemoryAssetRepository`的代码

```py
from itam.infrastructure.repository.base_repository import BaseRepository
from itam.domain.asset import Asset

class InMemoryAssetRepository(BaseRepository[Asset]):
    def __init__(self):
        self._assets = []

    def create(self, asset: Asset) -> None:
        self._assets.append(asset)

    def read(self, asset_id: int) -> Asset:
        return next((asset for asset in self._assets 
if asset.id == asset_id), None)

    def update(self, asset: Asset) -> None:
        for i in range(len(self._assets)):
            if self._assets[i].id == asset.id:
                self._assets[i] = asset
                break

    def delete(self, asset_id: int) -> None:
        self._assets = [asset for asset in self._assets 
if asset.id != asset_id]

    def get_entities(self) -> list[Asset]:
        return self._assets
```

最后，我们将更新`AssetManager`以将`Asset`的 CRUD 操作委托给一个`BaseRepository`实例（`_repository`）。以下是完整的源代码，包括文件开头的提示。

列表 4.28 使用`InMemoryAssetRepository`的`AssetManager`

```py
# Define a class called AssetManager
# with CRUD operations for Asset
# and a method to notify observers whenever an asset is created, updated or deleted
# The AssetManager should use an instance of the InMemoryAssetRepository class for data access and CRUD operations on Asset objects. 
# Implement methods for creating, reading, updating, and deleting assets using the AssetRepository instance. 
# Please include type hints for the input parameters and return values.
# The methods should be named create, read, update, get_assets, and delete.

from itam.domain.asset import Asset
from itam.infrastructure.repository.in_memory_asset_repository import InMemoryAssetRepository

class AssetManager:
    def __init__(self):
        self._repository = InMemoryAssetRepository()

    def create(self, asset: Asset) -> Asset:
        self._repository.create(asset)
        return asset

    def read(self, asset_id: int) -> Asset:
        return self._repository.read(asset_id)

    def update(self, asset: Asset) -> Asset:
        self._repository.update(asset)
        return asset

    def get_assets(self) -> list[Asset]:
        return self._repository.get_entities()

    def delete(self, asset_id: int) -> None:
        self._repository.delete(asset_id)
```

到目前为止，我们有一个核心业务领域，它没有直接被我们的系统直接触及。我们有端口供请求进入。我们也有端口可以存储数据（至少在系统运行时）。我们应该能够通过运行系统，向创建端点发送 POST 请求，并从 GET 端点读取数据来测试系统的端到端。一旦我们确认系统端到端工作正常，我们可以解决我们的数据只保存在内存中的问题；我们现在可以连接一个实际的数据库。为此，我们引入本章的最后一个模式：单例模式。从概念上讲，单例模式非常简单易懂；它应该只有一个实例在运行。它适用于许多用例：日志记录、缓存、配置设置或数据库连接管理器。

单例模式

单例模式是一种设计模式，确保一个类只有一个实例，并提供对该实例的全局访问点。当您想在程序的各个部分共享单个对象而不是创建相同类的多个实例时，会使用它。

### 4.3.4 集中（并外部化）我们的数据访问

我们将要求 Copilot 为我们创建一个单例类来管理数据库的连接。我们不应该在我们的源代码中硬编码用户名或密码（或任何连接细节），这不仅因为它是固有的不安全，而且因为这些细节很可能根据环境（开发与测试与生产）的不同而不同。因此，我们将要求 Copilot 接受这些值作为环境变量，并使用它们来配置连接：

```py
# Create a Python class called DatabaseConnection using the Singleton pattern to manage a single connection to a PostgreSQL database using SQLAlchemy. 
# The class should read the database username, password, and connection string from environment variables
# The environment variables should be named: DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, and DB_NAME
```

我们要求 Copilot 使用 SQLAlchemy，一个对象关系映射（ORM）工具，对数据库执行操作。Copilot 将巧妙地组装我们的源代码。`@staticmethod`关键字创建了一个属于类而不是类实例的方法。这个关键字用于获取`DatabaseConnection`类的实例。因为静态方法不能修改实例数据，所以它可以用来确保这个类只有一个实例在运行，因此是一个单例。构造函数使用字符串插值和环境变量初始化数据库连接。

列表 4.29 `DatabaseConnection`的实现
