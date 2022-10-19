
# 数据加载

## 主要内容

数据的加载主要包括以下几个方面：
- 数据集的格式转换，需要支持各种类型各种格式的数据，如图片、语音、文本、表格等
- 数据的采样和shuffle，可能面临分布式的挑战。
- 数据增强，会产生额外的数据
- 数据预处理，如图片事先进行黑白二值化等
- 数据分batch
- 数据加载到内存，并且进入锁页内存
- 数据加载到GPU
- 数据分发给不同的计算单元，并且不会重复，且支持分布式训练

## 相关的源代码

DistributedSampler

torch/utils/dataset.py

## 模型训练中的数据集

下面我们先看一个利用CIFAR10数据集进行模型训练的例子：

```Python

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

# Model
print('==> Building model..')
net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    scheduler.step()

```

在这个例子中，训练使用的是torch.utils.data.DataLoader，我们先从DataLoader入手，看看PyTorch是如何管理数据的。

当前业界普遍使用GPU进行模型训练，GPU的吞吐率很高，很容易导致数据的加载成为瓶颈。因此PyTorch的DataLoader支持多个worker同时加载数据。相应地，DataLoader的迭代器有两类：_SingleProcessDataLoaderIter处理单个worker的情况，而_MultiProcessingDataLoaderIter处理多个worker的情况。

因为是单进程处理，_SingleProcessDataLoaderIter的处理逻辑相对清晰，主要的工作是读取数据的Fetcher和pin_memory()这两部分。
<img src='../images/4.png'>

在多进程的情况下，最耗费时间的Fetcher部分和pin_memory()部分改成了多进程，如下图：
<img src='../images/3.png'>



```Python
#Harry  torch/utils/data/dataloader.py

class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: int
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)


```
### 设计原则1. DataLoader -> Dataset



## GIL

GIL(Global Interpreter Lock)是一个全局的锁，深度学习框架属于计算密集型的系统，在进行计算时要尽可能的

## 参考

- 一文详解 Python GIL 设计 https://zhuanlan.zhihu.com/p/363040263
- 万字综述，核心开发者全面解读PyTorch内部机制 https://zhuanlan.zhihu.com/p/67834038
- https://blog.csdn.net/u013608424/article/details/123782284