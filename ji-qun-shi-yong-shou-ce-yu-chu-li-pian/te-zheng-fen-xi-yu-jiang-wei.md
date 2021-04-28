---
description: 作者：唐元博         时间：2020-4-19
---

# 特征分析与降维

## 前言

根据[Outlin](outline.md)e中的分析，我们知道原始的大数据集中包含大和脏两大特点，在本页我们主要针对特征维度高这一特点作分析和处理，我们的目标是分析高维特征、并尝试在保留数据质量的前提下尽可能降低特征维度，达到降低数据量、便于后续程序分析的效果。

## 分析

首先，我们需要对数据的高维特征作分析，因为只有对于这些特征有相当清晰的认识，才可以作下一步的特征抽取或转换。在[数据统计](shu-ju-yu-chu-li-zhi-tong-ji-fen-xi.md)这一页中我们介绍了数据的基本统计操作，包括对于特征缺失率、均值、方差、最大最小值的统计，根据这些信息，再结合背景语义，我们可以大概对某一维特征的质量、规模等情况有一个基本的认识。

接下来，通过normalization方法我们可以将不同尺度的特征缩放到同一尺度下，通过计算特征之间的相关系数，我们可以知晓特征之间的关联性，为下一步降维作准备。下边的代码块提供了对数据进行normalize的操作，这样可以使得不同的特征保持在同样的分布，提高后续机器学习算法的性能。

```python
%pyspark
from pyspark.ml.feature import StandardScaler
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 1, 1.0]),),
    (1, Vectors.dense([0, -1.0, -1.0]),),
    (2, Vectors.dense([-1, 10.0, -6.0]),)
], ["id", "features"])

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
scaledData.show()
```

默认情况下StandradScalar函数会将每一列的feature变换成0均值，标准方差的特征，均值和方差均可设置，下图展示一个3维矩阵的变换结果，更高维的数据同理。

![&#x5BF9;&#x7279;&#x5F81;&#x8FDB;&#x884C;&#x6807;&#x51C6;&#x5316;](../.gitbook/assets/image%20%282%29.png)

## 降维

对高维数据降维有两种主要方式：特征选择和特征转换，前者意味着使用一部分相对重要的特征而丢弃那些不太重要的，后者则是将全体特征投影到一个新的空间中，通过去除特征之间的相关性进行数据压缩（PCA是其中典型的方法）。大体上前者的优势在于易于理解、解释，后者则有着更好的压缩性价比（即在同样信息损失的情况下压缩率大大增加）。

### 特征选择

卡方检验是特征选择中常用的算法之一，其最基本的思想就是通过观察实际值与理论值的偏差来确定理论的正确与否。具体做的时候常常先假设两个变量确实是独立的（行话就叫做“原假设”），然后观察实际值（也可以叫做观察值）与理论值（这个理论值是指“如果两者确实独立”的情况下应该有的值）的偏差程度，如果偏差足够小，我们就认为误差是很自然的样本误差，是测量手段不够精确导致或者偶然发生的，两者确确实实是独立的，此时就接受原假设；如果偏差大到一定程度，使得这样的误差不太可能是偶然产生或者测量不精确所致，我们就认为两者实际上是相关的，即否定原假设，而接受备择假设。如果希望更细节的了解这个方法可以参照以下链接

{% embed url="https://blog.csdn.net/idatamining/article/details/8564981" caption="" %}

然后我们在集群上简单实现了这个算法：

```python
%pyspark
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (7, Vectors.dense([1.1, 0.0, -1.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 11.0, 102.0, -0.1]), 0.0,),
    (9, Vectors.dense([0.0, 0.0, 125.0, 0.1]), 0.0,)], ["id", "features", "clicked"])
selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="clicked")
result = selector.fit(df).transform(df)
print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()
```

![&#x6839;&#x636E;&#x5361;&#x65B9;&#x68C0;&#x9A8C;&#x9009;&#x62E9;&#x51FA;&#x6765;&#x7684;&#x7279;&#x5F81;](../.gitbook/assets/image%20%2831%29.png)

### 特征转换

在特征转换算法中，主成分分析是相当经典和基础的方法，其思想主要是通过去除特征中的相关性达到在尽量减少数据规模的同时保留信息。关于PCA算法的介绍，可以参阅

{% embed url="https://zhuanlan.zhihu.com/p/77151308" caption="" %}

这里给出在spark平台上对数据进行PCA降维的示例代码，为了简便，我们采用一个3\*5的矩阵举例。代码运行在[http://10.129.2.159:9090](http://10.129.2.159:9090/)的zepplin平台上。

```python
#首先指定编译器为pyspark
%pyspark 
#导入相关包
from pyspark.ml.linalg import Vectors 
from pyspark.ml.feature import PCA 
#设定数据 并转化为spark的dataframe格式数据
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),), 
        Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),), 
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)] 
df = spark.createDataFrame(data,["features"]) 
#直接调用PCA函数，k=2意思是取前两个最大的主成分
pca = PCA(k=2, inputCol="features", outputCol="pca_features") 
#使用先前指定的data计算映射model
model = pca.fit(df) 
#将数据映射到低维空间，并打印第一维数据
model.transform(df).collect()[0].pca_features
#运行结果为
#DenseVector([1.6486, -4.0133])
```

## 其他参考链接

{% embed url="http://spark.apache.org/docs/latest/ml-features.html" caption="" %}

