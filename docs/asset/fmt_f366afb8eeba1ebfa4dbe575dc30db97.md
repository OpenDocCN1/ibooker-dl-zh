
> 以下代码定义了一个`CategoricalEncoder`类，该类用于将分类特征编码为数值数组。此代码源自PR #9151，可直接运行或复制到你的代码中，无需深入理解其实现细节（目前阶段）。

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """将分类特征编码为数值数组。

    此转换器的输入应为整数或字符串矩阵，表示分类（离散）特征的值。
    特征可以使用one-hot（即one-of-K）方案（``encoding='onehot'``，默认）编码，
    或转换为序数整数（``encoding='ordinal'``）。

    这种编码对于将分类数据输入到许多scikit-learn估计器（尤其是线性模型和标准核SVM）是必需的。
    更多信息请参阅:ref:`用户指南 <preprocessing_categorical_features>`。

    参数
    ----------
    encoding : str, 'onehot', 'onehot-dense' 或 'ordinal'
        要使用的编码类型（默认为'onehot'）：
        - 'onehot'：使用one-hot（即one-of-K）方案（也称为'dummy'编码）对特征进行编码。
          这会为每个类别创建一个二进制列，并返回一个稀疏矩阵。
        - 'onehot-dense'：与'onehot'相同，但返回密集数组而不是稀疏矩阵。
        - 'ordinal'：将特征编码为序数整数。这会导致每个特征生成一个整数列（0到n_categories - 1）。
    categories : 'auto' 或值列表/数组的列表。
        每个特征的类别（唯一值）：
        - 'auto'：从训练数据中自动确定类别。
        - list : ``categories[i]`` 保存第i列中预期的类别。在编码数据之前，传递的类别会被排序
          （使用的类别可以在``categories_``属性中找到）。
    dtype : 数字类型，默认np.float64
        输出的期望dtype。
    handle_unknown : 'error'（默认）或 'ignore'
        在转换期间遇到未知分类特征时是引发错误还是忽略（默认是引发错误）。
        当此参数设置为'ignore'且在转换期间遇到未知类别时，此特征的one-hot编码列将全为零。
        对于``encoding='ordinal'``，不支持忽略未知类别。

    属性
    ----------
    categories_ : 数组列表
        在拟合期间确定的每个特征的类别。当手动指定类别时，此属性保存排序后的类别
        （顺序与`transform`的输出相对应）。

    示例
    --------
    给定一个包含三个特征和两个样本的数据集，我们让编码器找到每个特征的最大值，
    并将数据转换为二进制one-hot编码。

    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])

    另请参阅
    --------
    sklearn.preprocessing.OneHotEncoder : 对整数序数特征执行one-hot编码。
      ``OneHotEncoder`` 假设输入特征的值在``[0, max(feature)]``范围内，而不是使用唯一值。
    sklearn.feature_extraction.DictVectorizer : 对字典项执行one-hot编码（也处理字符串值特征）。
    sklearn.feature_extraction.FeatureHasher : 对字典项或字符串执行近似one-hot编码。
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """将CategoricalEncoder拟合到X。

        参数
        ----------
        X : 数组形式，形状 [n_samples, n_feature]
            用于确定每个特征类别的数据。

        返回
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """使用one-hot编码转换X。

        参数
        ----------
        X : 数组形式，形状 [n_samples, n_features]
            要编码的数据。

        返回
        -------
        X_out : 稀疏矩阵或二维数组
            转换后的输入。
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # 将问题行设置为可接受的值，并继续
                    # `行被标记为`X_mask`，稍后将被移除。
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
```
