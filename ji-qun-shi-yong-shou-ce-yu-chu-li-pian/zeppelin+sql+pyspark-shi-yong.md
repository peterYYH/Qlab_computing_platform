---
description: 作者：杨煜涵         时间：2020-4-19
---

# Zeppelin+sql+pyspark使用

## 资源性能对比

### zeppelin资源配置

zeppelin这个工具非常方便的嵌入了多种编译器，具有数据可视化功能，以及编译器的资源配置功能。相较之前计算资源的通过shell命令的分配，zeppelin可以直接通过web页面指定资源的配置：

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2847%29.png)

### 多核与单核性能对比

实验中，我们选取了13G的数据展示一下多核并行处理计算速度的优势。（需要注意的是，并不是所有计算都能达到加速的效果，对于一些小规模数据集，或者是算法中存在着非常复杂的迭代递归计算，spark并不一定适合），**本次实验主要对比的是cpu核数对计算速度的影响**：

数据描述：纽约311服务热线采集的用户数据，有41个索引标签，共有21960000条记录，文件大小13.68 GB

| code | core 1, memory 50G | core 20, memory 50G | 速度提升 |
| :--- | :--- | :--- | :--- |
| 计算缺失率 | 4 min 50 sec | 34 sec | 8倍 |
| 数据筛选filter | 2 min 21 sec | 22 sec | 6倍 |
| distinct\(\).count\(\) | 2 min 24 sec | 13 sec | 11倍 |
| 统计缺失个数 | 2 min 22 sec | 14 sec | 10倍 |
| 统计数量并排序 | 2 min 28 sec | 11 sec | 13倍 |
| groupBy && count | 2 min 26 sec | 12 sec | 12倍 |

可以明显的发现，对于大规模数据的普通计算统计，多核效率提升明显，同时spark有大量内置的API函数，我们只需编写串行代码，机器自动并行执行，开发方便

## 数据可视化

zeppelin这个web工具，对于sql的可视化支持的非常好，我们只需将csv文件读入即可进行，然后生成临时表，即可编写sql语句操作，zeppelin会自动生成可视化的图片或者表格（python需要自己利用matplotlib包来绘图）：

{% tabs %}
{% tab title="Scala" %}
```scala
%spark
val df=spark.read.option("header","true").csv("hdfs://10.129.2.155:50090/123/data/311-data/311-service-requests-from-2010-to-present.csv")
df.groupBy("City").count().createOrReplaceTempView("city")
```

```scala
%sql
select * from city
```
{% endtab %}
{% endtabs %}

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2822%29.png)

![](../.gitbook/assets/image%20%2812%29.png)

zeppelin的sql编译器还提供了交互式的操作，可以通过图形界面选择自己需要的 关键字和值 做统计分析：

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2819%29.png)

