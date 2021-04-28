---
description: 作者：杨煜涵           时间：2020-4-22
---

# spark数据结构

![RDD \(Spark1.0\) &#x2014;&amp;gt; Dataframe\(Spark1.3\) &#x2014;&amp;gt; Dataset\(Spark1.6\)](../.gitbook/assets/image%20%2815%29.png)

## RDD（Resilient Distributed Datasets）

an RDD is a read-only, partitioned collection of records. 字面意思是只读的分布式数据集。类似于关系数据库 里的一个个操作，比如 map，filter，Join 等。在 Spark 里面实现了许多这样的 RDD 类，即可以看成是操作类。当我们调用一个 map 接口，底层实现是会生成一个 MapPartitionsRDD 对象，当 RDD 真正执行时，会调用 MapPartitionsRDD 对象里面的 compute 方法来执行这个操作的计算逻辑。但是不同的是，RDD 是 lazy 模式，只有像 count，saveasText 这种 action 动作被调用后再会去触发 runJob 动作。

RDD 分为二类：transformation 和 action。

* transformation 是从一个 RDD 转换为一个新的 RDD 或者从数据源生成一个新的 RDD；
* action 是触发 job 的执行。所有的 transformation 都是 lazy 执行，只有在 action 被提交的时候才触发前面整个 RDD 的执行

### 适用场景

* 你需要使用low-level的transformation和action来控制你的数据集； 
* 你的数据集非结构化，比如：流媒体或者文本流； 
* 你想使用函数式编程来操作你的数据，而不是用特定领域语言\(DSL\)表达； 
* 你不在乎schema，比如，当通过名字或者列处理\(或访问\)数据属性不在意列式存储格式； 
* 你放弃使用DataFrame和Dataset来优化结构化和半结构化数据集

## DataFrame

DataFrame与RDD相同之处，都是不可变分布式弹性数据集。不同之处在于，**DataFrame的数据集都是按指定列存储，即结构化数据**。类似于传统数据库中的表。DataFrame的设计是为了让大数据处理起来更容易。DataFrame允许开发者把结构化数据集导入DataFrame，并做了higher-level的抽象；DataFrame提供特定领域的语言\(DSL\)API来操作你的数据集。除了数据以外，还记录数据的结构信息，即schema。同时，与Hive类似，DataFrame也支持嵌套数据类型（struct、array和map）。从API易用性的角度上看，DataFrame API提供的是一套高层的关系操作，比函数式的RDD API要更加友好，门槛更低。由于与R和Pandas的DataFrame类似，Spark DataFrame很好地继承了传统单机数据分析的开发体验。

![](https://github.com/peterYYH/Qlab_computing_platform/tree/980bb64ec50ce521e87d75d7773895a118e920fb/.gitbook/assets/image%20%2838%29.png)

上图直观地体现了DataFrame和RDD的区别。左侧的RDD\[Person\]虽然以Person为类型参数，但Spark框架本身不了解Person类的内部结构。而右侧的DataFrame却提供了详细的结构信息，使得Spark SQL可以清楚地知道该数据集中包含哪些列，每列的名称和类型各是什么。DataFrame多了数据的结构信息，即schema。RDD是分布式的Java对象的集合。DataFrame是分布式的Row对象的集合。DataFrame除了提供了比RDD更丰富的算子以外，更重要的特点是提升执行效率、减少数据读取以及执行计划的优化，比如filter下推、裁剪等。**性能上比RDD要高 。**

DataFrame和Dataset API获得空间效率和性能优化的两个原因：

* ataFrame和Dataset API是建立在Spark SQL引擎之上，它会使用Catalyst优化器来生成优化过的逻辑计划和物理查询计划
* R，Java，Scala或者Python的DataFrame/Dataset API使得查询都进行相同的代码优化以及空间和速度的效率提升。

![](../.gitbook/assets/image%20%2827%29.png)

Dataset和DataFrame拥有完全相同的成员函数，区别只是每一行的数据类型不同。DataFrame也可以叫Dataset\[Row\],每一行的类型是Row，不解析，每一行究竟有哪些字段，各个字段又是什么类型都无从得知，只能用上面提到的getAS方法或者模式匹配拿出特定字段.而Dataset中，每一行是什么类型是不一定的，在自定义了case class之后可以很自由的获得每一行的信息

## **Dataset**

在Spark 2.0中，Dataset具有两个完全不同的API特征：强类型API和弱类型API，见下表。**DataFrame是特殊的Dataset**，其每行是一个弱类型JVM object。相对应地，Dataset是强类型JVM object的集合，通过Scala的case class或者Java class。

![](../.gitbook/assets/image%20%2818%29.png)

### DataFrame或者Dataset适用场景

* 你想使用丰富的语义，high-level抽象，和特定领域语言API
* 你处理的半结构化数据集需要high-level表达，filter，map，aggregation，average，sum，SQL查询，列式访问和使用lambda函数 
* 你想利用编译时高度的type-safety，Catalyst优化和Tungsten的code生成
* 你想统一和简化API使用跨Spark的Library
* 如果你是一个R使用者
* 如果你是一个Python使用者，那你可以使用DataFrame或者Dataset。 你可以无缝地把DataFrame或者Dataset转化成一个RDD，只需简单的调用.rdd

## 总结

### RDD：

* RDD适合需要low-level函数式编程和操作数据集的情况；
* **RDD一般和spark mlib同时使用**
* RDD不支持sparksql操作

### dataframe && dataset

* DataFrame和Dataset适合结构化数据集，使用high-level和特定领域语言\(DSL\)编程，空间效率高和速度快。
* 与RDD和Dataset不同，DataFrame每一行的类型固定为Row，只有通过解析才能获取各个字段的值,每一列的值没法直接访问
* DataFrame与Dataset一般不与spark ml同时使用
* **DataFrame与Dataset均支持sparksql的操作**，比如select，groupby之类，还能注册临时表/视窗，进行sql语句操作
* DataFrame与Dataset支持一些特别方便的保存方式，比如保存成csv，可以带上表头，这样每一列的字段名一目了然

![](../.gitbook/assets/image%20%288%29.png)

