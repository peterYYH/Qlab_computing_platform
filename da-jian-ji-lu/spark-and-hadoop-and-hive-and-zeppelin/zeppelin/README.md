---
description: 作者：杨煜涵     时间：4月22日
---

# Zeppelin可视化数据分析工具

## 简介

[Zeppelin](https://blog.csdn.net/laozhaokun/article/details/44803061)是一个Apache的孵化项目。一个基于web的笔记本，支持交互式数据分析。你可以用SQL、Scala等做出数据驱动的、交互、协作的文档。\(类似于ipython notebook，可以直接在浏览器中写代码、笔记并共享\)

## 下载与安装

### 下载地址：[http://zeppelin.apache.org/](http://zeppelin.apache.org/)

### 安装：

#### 1.解压后进入conf文件夹，复制两个模板文件

`~/conf$ scp zeppelin-site.xml.template zeppelin-site.xml`

`~/conf$ scp zeppelin-env.sh.template zeppelin-env.sh`

#### 2.配置 zeppelin-env.sh

`hadoop@slave4:~/local/zeppelin-0.8.0-bin-all/conf$ vim zeppelin-env.sh`

添加java和spark的路径，因为是集群模式，需要设置Master（默认为localhost模式）

```text
export JAVA_HOME=~/local/jdk1.8.0_221
export MASTER=spark://slave1:7077
export SPARK_HOME=~/local/sparkwithhive
export HADOOP_CONF_DIR=~/local/hadoop-2.9.0
```

#### 3.配置zeppelin-site.xml

这个配置是配置端口号，默认的端口是localhost:8080,因为8080常用，尽量更改

`hadoop@slave4:~/local/zeppelin-0.8.0-bin-all/conf$ vim zeppelin-site.xml`

```text
<property>
  <name>zeppelin.server.addr</name>
  <value>10.129.2.159</value>
  <description>Server address</description>
</property>

<property>
  <name>zeppelin.server.port</name>
  <value>9090</value>
  <description>Server port.</description>
</property>
```

#### 4.打开网络

需要注意这一步不能省！第一次启动需要连到aws的服务器，否则会启动失败，打不开网页

`hadoop@slave4:~$ python netLogin3.py`

#### 5.启动服务

{% hint style="info" %}
注意启动必须要用sudo，否则网页打不开，同时不能切换到root用户，zeppelin打开依赖java环境，root用户还未配置java环境。

同时建议用zeppelin-daemon.sh start而不用zeppelin.sh
{% endhint %}

```text
#进入bin文件夹
~/bin$ sudo ./zeppelin-daemon.sh start
```

#### 6.验证

```text
# 切换到root用户
~/bin$ sudo su
# 查看服务状况
~/bin$ jps
```

![&#x542F;&#x52A8;&#x6210;&#x529F;](../../../.gitbook/assets/image%20%2820%29.png)

#### 7.进入zeppelin

浏览器进入网页号：[10.129.2.159:9090](http://10.129.2.159:9090/#/) 页面中有 **zeppelin tutorial** 可供学习

## 问题

1.首次启动必须要在**联网**的情况下，否则web页面

2.下载的文件包conf文件夹下不存在interpreter.json文件，首次安装会生成此文件，但可能会存在权限问题，用户为root，如果存在此情况可能打不开网页，需要将interpreter.json用户改成hadoop

`~/bin$ sudo chown hadoop:hadoop interpreter.json`

3.版本冲突：  
若zeppelin和spark的版本冲突，spark中的三个包替换到zeppelin中

## [下一章节具体介绍zeppelin的使用方法以及编译器配置](zeppelin-pei-zhi-yu-shi-yong.md)

