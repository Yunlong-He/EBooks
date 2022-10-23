# 分布式

## 本章主要内容
- 为什么需要分布式
- 分布式的难点在哪里？
- PyTorch中的相关模块
    - THD
    - C10D
    - torch.multiprocessing
    - torch.distributedDataParallel（DP）
    - DistributedDataParallel（DDP）
    - torch.distributed.rpc

## 什么是分布式训练

###分布式计算
由于单个节点的计算能力有限，对于计算密集型的任务，只在单个节点上运行，可能会花费非常多的时间，此时充分利用多个节点协作完成任务是最合适的选择。
将任务从单节点转化为分布式任务，需要考虑不同节点间的通信，包括输入数据的拆分，临时数据的分发与归并，计算结果的合并，以及计算过程中的同步控制等等，而这些因素由于任务类型的多样化，也会变得非常复杂，没有完美的方案可以处理不同的情况，因此在特定类型的任务下采用不同的解决方案也是必然的。

为了简化算法开发的复杂度，将分布式计算中的数据分发和网络通信与具体的算法应用分开，先驱们开发了不同的分布式计算框架，应用较广的包括MPI、MapReduce、Spark等，在科学研究领域的高性能计算、互联网时代的海量数据处理和分析场景中，这些框架已经成为不可替代的基础软件。

在深度学习领域，模型的效果主要来自于两个方面：海量的数据和精心设计的复杂网络结构，这两点使得深度学习模型训练的计算复杂度很高，而且随着近些年超大模型取得了令人惊叹的效果，这个趋势也随之愈演愈烈，如下图：

<img src='../images/big_model_trends.webp'/>
<font size=2 style='italic'>来源：Compute Trends Across Three Eras of Machine Learning</font>

### 深度学习模型分布式训练的进展


### PyTorch中的分布式训练

## 参考
- https://zhuanlan.zhihu.com/p/136372142

