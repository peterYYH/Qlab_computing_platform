---
description: 作者：杨煜涵
---

# jupyter && pyspark联合使用

不习惯zeppelin网页工具的，可以用jupyter notebook直接配置使用pyspark，效果一样。**推荐使用方式1**

## 使用方式1：（推荐！！）

### jupyter中导入pyspark的包使用

默认anaconda或者jupyter notebook已经安装好

**主要是将pyspark中的一些包识别到路径中，可以直接调用**

step1: 打开jupyter, 映射已经设置好，端口号10.129.2.155:8899

```text
hadoop@slave1:~$ jupyter notebook
```

step2: 新建ipynb文件，输入\(**推荐方式1**\)

{% tabs %}
{% tab title="方式1" %}
```text
# 借助findspark包识别spark路径
import findspark
findspark.init()

# 直接启动sparkContext
import pyspark
sc=pyspark.SparkContext(master="spark://slave1:7077",appName="yyh_test1")

## 使用sparkSession spark2.0以后用的
from pyspark.sql import SparkSession
# sc=SparkSession.builder\
.appName("new_spark")\
.config("spark.executor.memory","10g")\
.config("spark.executor.cores","8")\
.getOrCreate()
```
{% endtab %}

{% tab title="方式2" %}
```text
import os
import sys
spark_name = os.environ.get('SPARK_HOME',None)
if not spark_name:
    raise ValueErrorError('spark环境没有配置好')
sys.path.insert(0,os.path.join(spark_name,'python'))
sys.path.insert(0,os.path.join(spark_name,'python/lib/py4j-0.10.4-src.zip'))
# exec(open(os.path.join(spark_name,'python/pyspark/shell.py')).read())
```
{% endtab %}
{% endtabs %}

[参考链接](https://blog.csdn.net/dxyna/article/details/79772343)

## 使用方式2：\(启动有点麻烦\)

### 直接在命令行中启动pyspark，设置启动选项为jupyter

在命令行中输入：

```text
PYSPARK_DRIVER_PYTHON=~/anaconda3/bin/jupyter-notebook pyspark
```

```text
master$ PYSPARK_DRIVER_PYTHON=~/anaconda3/bin/jupyter-notebook PYSPARK_DRIVER_PYTHON_OPTS=" 
--ip=0.0.0.0 --port=7777" pyspark 
--packages com.databricks:spark-csv_2.11:1.1.0 
--master spark://spark_master_hostname:7077 
--executor-memory 6400M --driver-memory 6400M
```

## 使用方式3：直接在环境变量中添加（不推荐）

**step1:**

```text
vim ~/.bashrc
```

**step2: 添加如下信息**  
`export PYSPARK_DRIVER_PYTHON=jupyter    
export PYSPARK_PYTHON=/home/hadoop/anaconda3/bin/python3    
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"`  
**step3:**  
`source ~/.bashrc`  
**step4: 启动**  
运行pyspark直接启动

```bash
hadoop@slave1:~$ pyspark
```

或者指定一些：

```text
pyspark --master spark://slave:7077 --executor-memory xxxM --total-executor-cores xx
```

### 缺点：

因为export了`PYSPARK_DRIVER_PYTHON` 与`PYSPARK_DRIVER_PYTHON_OPTS` 两个环境变量后，非shell的pyspark 生怕认可应用也将使用jupyter-notebook，这必然引起混乱，**所以推荐方法一和方法二**  
实测，添加到环境变量中后，spark只支持ipynb的文件，因此不用。

[参考链接](https://blog.csdn.net/NJZhuJinhua/article/details/79441217)

## 如下是常用的启动命令

```text
spark-submit --master spark://slave1:7077 --total-executor-cores 144 
--executor-memory 70g --driver-memory 30g
```

