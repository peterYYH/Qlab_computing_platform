---
description: 参与人员：唐元博、杨煜涵、陈靖仪          时间：2020年3月15日
---

# 使用kubernetes加速covid-19模型训练

1 概述 

2 具体使用步骤 

2.1 程序容器化打包 

2.2 将运行数据与程序上传至分布式文件系统中 

2.3 编写kubernetes文件 

2.4 提交任务队列 

## 1.概述

在日常工作科研中我们经常会遇到这样的一些计算任务：批处理类型，可以很容易拆分成为很多个相互之间没有数据依赖的小型任务。本文档提出了一种在实验室Kubernetes集群进行多任务并行的一种加速方式。任务执行主要结构如下：

![&#x56FE; 1 kubernetes&#x6279;&#x5904;&#x7406;&#x793A;&#x610F;&#x56FE;](../.gitbook/assets/0%20%281%29.png)

## 2.具体使用步骤

### 2.1程序容器化打包

顾名思义，是将原本运行在本地的程序移植到容器中。容器具有隔离性强、轻量型等优势，是执行大量小型任务的理想选择。将本地的程序移植到容器中并不困难，仅需拉取一个ubuntu等常见容器，在其中配置一下环境即可（如python包），在这里我们以ubuntu/hadoop-yyh:latest这个镜像作为示例。

### 2.2将运行数据与程序上传至分布式文件系统中

为了使得所有批处理任务得以相互通信，我们采用每一个任务都可以访问的分布式文件系统hdfs作为共有数据中转。在运行任务之前将需要运行的代码和数据上传至hdfs中，每一个任务启动时，首先从hdfs上下载程序和自己对应的文件，运算之后将结果保存至hdfs，最后再进行结果的整合即可。

![&#x56FE; 2 &#x5206;&#x5E03;&#x5F0F;&#x6587;&#x4EF6;&#x7CFB;&#x7EDF;&#x7684;web&#x754C;&#x9762;](../.gitbook/assets/1%20%284%29.png)

在图2的web界面中可以很方便的进行数据和代码的上传下载。

### 2.3编写kubernetes文件

为了在kubernetes集群上执行批处理任务，需要向kubernetes的主节点提交一份表单，这张表单上描述了使用什么镜像、容器启动之后依次执行的命令序列。

![&#x56FE; 3 &#x8868;&#x5355;&#x793A;&#x4F8B;](../.gitbook/assets/2%20%282%29.png)

在图3中我们展示了一个yaml的示例，其中指定了image的信息，而红色字体每一行代表容器启动之后依次执行的命令，可以看到它就是常用的linux命令。

### 2.4提交任务队列

![&#x56FE; 4 &#x63D0;&#x4EA4;&#x4EFB;&#x52A1;&#x961F;&#x5217;&#x793A;&#x4F8B;](../.gitbook/assets/3%20%281%29.png)

在图四中我们看到，使用kubectl apply命令之后，jobs下的所有任务均被提交。

![&#x56FE; 5 &#x5206;&#x5E03;&#x5F0F;&#x6587;&#x4EF6;&#x7CFB;&#x7EDF;&#x7ED3;&#x679C;](../.gitbook/assets/4%20%281%29.png)

一段时间之后，结果均被保存在hdfs上，在156服务器上也有一个代码用以将单个任务的结果整合到一起，至此，并行的批处理执行演示完毕。
