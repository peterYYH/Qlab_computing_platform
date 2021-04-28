---
description: 作者：陈靖仪           时间：2020-4-22
---

# 数据清洗

原始数据是未经处理的，可能存在着大量重复数据，缺失数据和异常数据。在对数据统计或建模之前，必须先进行数据清洗的工作。

## 重复数据

数据中可能存在某些记录的特征完全相同的重复现象，如果数据是以ID形式来区分每条记录，可能出现三种情况：1.ID相同且其特征也完全相同；2.ID不同但除ID外所有特征相同3. 特征不同但ID相同

首先，先检查第一种数据完全相同的重复情况：比较完整数据集和运行.distinct\(\)方法后的数据集的数量。

```python
#创建以下示例数据：
df = spark.createDataFrame([
 (1,144.5,5.9,33,'M'),
 (2,167.2,5.4,45,'M'),
 (3,124.1,5.2,23,'F'),
 (4,144.5,5.9,33,'M'),
 (5,133.2,5.7,54,'F'),
 (3,124.1,5.2,23,'F'),
 (5,129.2,5.3,42,'M'),
 ],['id','weight','height','age','gender'])
```

检查重复值，两个数字不同，则认为有完全相同的重复数据。可以使用.dropDuplicates\(…\)方法将重复行移除。

```python
print('Count of rows:{0}'.format(df.count()))
print('Count of distinct rows:{0}'.format(df.distinct().count()))
df=df.dropDuplicates()
```

![](../.gitbook/assets/image%20%284%29.png)

然后，检查第二种重复情况，对ID列以外的列作比较

{% tabs %}
{% tab title="Python" %}
```python
print('Count of ids:{0}'.format(df.count()))
print('Count of distinct ids:{0}'.format(
 df.select([c for c in df.columns if c!='id']).distinct().count())
)
#引入subset参数，只查找subset指定的列，以移除ID列以外内容相同的记录
df=df.dropDuplicates(subset=[c for c in df.columns if c!='id'])
```
{% endtab %}
{% endtabs %}

![](../.gitbook/assets/image%20%287%29.png)

接着，使用.count\(…\)和.countDistinct\(…\)计算DataFrame的行数和ID 唯一数。

因为已经移除所有重复数据，而ID数量却小于总数量，说明ID存在重复值，可用.monotonically\_increasing\_id\(\)方法给每一条记录提供唯一且递增的ID

```python
import pyspark.sql.functions as fn
df.agg(fn.count('id').alias('count'),
 fn.countDistinct('id').alias('distinct')
 ).show()
df.withColumn('new_id',fn.monotonically_increasing_id()).show()
```

![](../.gitbook/assets/image%20%2836%29.png)

## 缺失值

数据中经常出现某些特征值缺失的现象，对缺失值的处理可以通过直接移除数据方法实现，但是移除可能会对数据集的可能性造成严重的影响。检查数据每个特征的缺失率，如果一个特征的大部分值都是缺失的，则基本此特征无用，可以直接移除。

```python
#创建以下示例数据：
df_miss = spark.createDataFrame([
 (1, 143.5, 5.6, 28, 'M', 100000),
 (2, 167.2, 5.4, 45, 'M', None),
 (3, None , 5.2, None, None, None),
 (4, 144.5, 5.9, 33, 'M', None),
 (5, 133.2, 5.7, 54, 'F', None),
 (6, 124.1, 5.2, None, 'F', None),
 (7, 129.2, 5.3, 42, 'M', 76000)
 ], ['id', 'weight', 'height', 'age', 'gender', 'income'])
```

检查缺失值

```python
df_miss.agg(*[(1-fn.count(c)/fn.count('*')).alias(c+'_missing')
 for c in df_miss.columns]).show()
#移除大部分值都是缺失的特征
df_miss_no_income = df_miss.select([c for c in df_miss.columns if c!='income'])
#移除缺失值数量大于一定阈值的行
df_miss_no_income.dropna(thresh=3).show()
```

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2829%29.png)

另外处理缺失值的方法是对其做填充。如果数据时离散型布尔值，可以通过添加第三个类型——Missing，将其转换为另一个分类变量；如果数据是数值类型，可以填充任何的平均数、中间数或者其他预定义的值。

```python
#均值填充
means = df_miss_no_income.agg(*[fn.mean(c).alias(c)
 for c in df_miss_no_income.columns if c!= 'gender']
).toPandas().to_dict('records')[0]
means['gender']='missing'
df_miss_no_income.fillna(means).show()
```

![](../.gitbook/assets/image%20%2816%29.png)

## 离群值

异常数据是指那些与其他数据的分布明显偏离的数据。在最普遍的形式中，如果所有值大致在Q1-1.5IQR和Q3+1.5IQR范围内，IQR指四分位范围，可以认为没有离群值；IQR定义为上分为和下分位之差，也就是分别为第75个百分位（Q3）和第25个百分位（Q1）。

```python
#创建以下示例数据：
df_outliers = spark.createDataFrame([
(1, 143.5, 5.3, 28),
(2, 154.2, 5.5, 45),
(3, 342.3, 5.1, 99),
(4, 144.5, 5.5, 33),
(5, 133.2, 5.4, 54),
(6, 124.1, 5.1, 21),
(7, 129.2, 5.3, 42),
], ['id', 'weight', 'height', 'age'])
```

检查离群值，使用.approxQuantile\(…\)方法计算每个特征的上下截断点

```python
cols = ['weight','height','age']
bounds = {}
for col in cols:
 #计算数值型特征的上下截断点
 quantiles = df_outliers.approxQuantile(col,[0.25,0.75],0.05)
 #IQR为上分位与下分位之差
 IQR = quantiles[1] - quantiles[0]
 #设置取值范围
 bounds[col] = [
 quantiles[0] - 1.5*IQR,
 quantiles[1] + 1.5*IQR
 ]
```

替换离群值

```python
import pyspark.sql.functions as fn
#仅保留落在区间范围内的特征值，而离散值则变为null
no_outliers = df_outliers.select(*['id']+[(
fn.when(df_outliers[c].between(bounds[c][0], bounds[c][1]),df_outliers[c] )
).alias(c) for c in cols
])
no_outliers.show()
```

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2830%29.png)

此时，离群值可当做缺失值来做处理。

## 处理实例

使用数据：纽约311服务热线采集的用户数据，共有21960000条记录，文件大小13.68 GB

```python
import pyspark.sql.functions as fn
import pandas as pd
#读入存放在分布式文件系统中的数据
df=spark.read.csv("hdfs://10.129.2.155:50090/123/data/311-data/311-service-requests-from-2010-to-present.csv",header=True)
#统计重复率
print('Count of rows:{0}'.format(df.count()))
print('Count of distinct rows:{0}'.format(df.distinct().count()))
df=df.dropDuplicates()

print('Count of ids:{0}'.format(df.count()))
print('Count of distinct ids:{0}'.format(
    df.select([c for c in df.columns if c!='Unique Key']).distinct().count())
    )
#引入subset参数，只查找subset指定的列，以去掉id不同而内容相同的记录
df=df.dropDuplicates(subset=[c for c in df.columns if c!='Unique Key'])

total=fn.count('Unique Key')
df.agg(total.alias('count'),(fn.countDistinct('Unique Key')).alias('duplicates')).show()
```

```python
#统计缺失值,显示缺失率
df_miss=df.agg(*[
    (1-fn.count(c)/(fn.count('*'))).alias(c )
    for c in df.columns
])
pandas_df_miss = df_miss.toPandas()
conv = pandas_df_miss.T
conv = conv.reset_index()
conv.columns = ['attribute','missing_rate']
spark_df = spark.createDataFrame(conv)
spark_df.createOrReplaceTempView("missing")
```

通过Zepplin网页可视化的功能可以直观地查看各个属性的缺失率情况

![](../.gitbook/assets/image%20%2823%29.png)

由图可见，有些属性的缺失率过高，有些甚至达到了100%，可以认为这些属性对结果的影响可忽略不计，所以可以设置一个阈值，直接滤除缺失率达到指定阈值的属性

```python
#滤掉缺失率超过80%的属性
#pandas的dataframe转置操作，便于滤除
conv = pandas_df_miss.T
conv.columns = ['missing_rate']
conv=conv[conv.missing_rate<0.8]
spark_df = spark.createDataFrame(conv.T)

#选择符合条件的列
df=df.select([c for c in spark_df.columns])

#填充缺失值
new_df=df.na.fill('Unkown')
new_df.show()
```

