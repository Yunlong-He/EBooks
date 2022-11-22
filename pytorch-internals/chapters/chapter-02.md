# 初识PyTorch

## 主要内容
- [PyTorch的设计原则](#PyTorch的设计原则)
- [PyTorch的整体架构](#PyTorch的整体架构)
- [PyTorch的源代码结构](#PyTorch的源代码结构)
- [第三方依赖](#第三方依赖)


## PyTorch的设计原则

为了让开发者能够充分利用硬件的计算能力，同时保持很好的开发效率，PyTorch提供了功能丰富但设计优雅的Python API，并且将繁重的工作交由C++实现。C++部分是以Python扩展的形式工作的。

PyTorch的算子实现在ATen库中。

为了实现自动微分，在ATen之上，PyTorch增加了AutoGrad框架。

### C++扩展
在Python API层，PyTorch早期有Variable和Tensor两种类型，在v0.4.0之后，合并成了现在的Tensor类。

Python层的Tensor对象，到了C++层面用THPVariable来表示，实际C++层面使用时会先转换成C++的Tensor类型。
THP的含义： TH来自一<font color=red>T</font>orc<font color=red>H</font>，P指的是<font color=red>P</font>ython。

### numpy
考虑到numpy的广泛使用，PyTorch支持Tensor与numpy的互相转换。

### 零拷贝（zero-copy）

在机器学习和深度学习的场景中，tensor代表数据、权重、以及计算结果等，因此经常会出现带有大量数据的tensor。PyTorch中存在大量的tensor创建、计算、转换等场景，如果每一次都重新拷贝一份数据，会带来巨大的内存浪费和性能损失。，因此PyTorch支持零拷贝技术以减少不必要的消耗。如：
```Python
>>> np_array
   array([[1., 1.],
      [1., 1.]])
>>> torch_array = torch.from_numpy(np_array)
>>> torch_array.add_(1.0)
>>> np_array
   array([[2., 2.],
      [2., 2.]])
```
上面这种tensor和numpy array共用数据的操作，我们称为in-place操作，但有时候in-place操作和普通拷贝数据的操作（standard操作）的界限并不是很清楚，需要仔细甄别。

另外需要说明的是，tensor中的数据由Storage对象进行管理，这样就把tensor的元信息与其真正的数据存储解耦了。

### JIT
PyTorch是基于动态图范式的，开发者可以像写普通程序一样，在网络的执行中加上各种条件分支语句，这样做带来的好处是容易理解，方便调试，但是同时也会有效率的影响，尤其是模型训练好之后用于推理，此时模型基本不需要调试，分支条件基本也固定了，这时候效率反而是需要优先考虑的因素了。另外虽然开发的时候使用Python会提高开发效率，但推理的时候需要支持其他的语言及环境，此时需要把模型与原来的Python代码解耦。

对此，PyTorch在v1.0的版本中引入了torch.jit，支持将PyTorch模型转为可序列化及可优化的格式。作为Python静态类型的子集，TorchScrip也被引入到PyTorch中。


## PyTorch的整体架构

## PyTorch的源代码结构

```bash


pytorch
|--- android        # PyTorch for Android， 开发库及运行环境
|--- aten           # 主要是C++ Tensor库的实现
|--- benchamarks    # 对PyTorch进行Benchmarking的代码
|--- binaries       # 用于移动端的厕所
|--- c10            # 核心Tensor实现，支持服务器端和移动端
|--- caffe2         # 从Caffe2项目迁移过来的代码
|--- cmake          # PyTorch的整体编译脚本
|--- docs           # PyTorch文档工具，可以根据Python和C++代码生成对应的文档
|--- ios            # PyTorch for iOS
|--- modules        # 
|--- mypy_plugins   # 
|--- scripts        # 
|--- submodules     # 
|--- test           # 
|--- third_party    # 第三方库
|--- tools          # 
|--- torch          # PyTorch的Python接口
|--- torchgen       # 

torch
|--- csrc       # Torch C++ 扩展模块的实现代码
      |--- module.cpp       # Torch C++ 扩展模块的初始化及入口代码

```

### C10
C10，来自于Caffe Tensor Library的缩写。这里存放的都是最基础的Tensor库的代码，可以运行在服务端和移动端。PyTorch目前正在将代码从ATen/core目录下迁移到C10中。C10的代码有一些特殊性，体现在这里的代码除了服务端外还要运行在移动端，因此编译后的二进制文件大小也很关键，因此C10目前存放的都是最核心、精简的、基础的Tensor函数和接口。

C10目前最具代表性的一个class就是TensorImpl了，它实现了Tensor的最基础框架。继承者和使用者有：

```C++
Variable的Variable::Impl
SparseTensorImpl
detail::make_tensor<TensorImpl>(storage_impl, CUDATensorId(), false)
Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>
```

值得一提的是，C10中还使用/修改了来自llvm的SmallVector，在vector元素比较少的时候用以代替std::vector，用以提升性能; 

### ATen
ATen，来自于 A TENsor library for C++11的缩写；PyTorch的C++ tensor library。ATen部分有大量的代码是来声明和定义Tensor运算相关的逻辑的，除此之外，PyTorch还使用了aten/src/ATen/gen.py来动态生成一些ATen相关的代码。ATen基于C10，Gemfield本文讨论的正是这部分；

### Caffe2
为了复用，2018年4月Facebook宣布将Caffe2的仓库合并到了PyTorch的仓库,从用户层面来复用包含了代码、CI、部署、使用、各种管理维护等。caffe2中network、operators等的实现，会生成libcaffe2.so、libcaffe2_gpu.so、caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so（caffe2 CPU Python 绑定）、caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so（caffe2 CUDA Python 绑定），基本上来自旧的caffe2项目)

### Torch

Torch，部分代码仍然在使用以前的快要进入历史博物馆的Torch开源项目，比如具有下面这些文件名格式的文件：

``` Bash
TH* = TorcH
THC* = TorcH Cuda
THCS* = TorcH Cuda Sparse (now defunct)
THCUNN* = TorcH CUda Neural Network (see cunn)
THD* = TorcH Distributed
THNN* = TorcH Neural Network
THS* = TorcH Sparse (now defunct)
THP* = TorcH Python
```

PyTorch会使用tools/setup_helpers/generate_code.py来动态生成Torch层面相关的一些代码，这部分动态生成的逻辑将不在本文阐述，你可以关注Gemfield专栏的后续文章。

### 第三方依赖

```bash
#Facebook开源的cpuinfo，检测cpu信息的
third_party/cpuinfo

#Facebook开源的神经网络模型交换格式，
#目前Pytorch、caffe2、ncnn、coreml等都可以对接
third_party/onnx

#FB (Facebook) + GEMM (General Matrix-Matrix Multiplication)
#Facebook开源的低精度高性能的矩阵运算库，目前作为caffe2 x86的量化运算符的backend。
third_party/fbgemm

#谷歌开源的benchmark库
third_party/benchmark

#谷歌开源的protobuf
third_party/protobuf

#谷歌开源的UT框架
third_party/googletest

#Facebook开源的面向移动平台的神经网络量化加速库
third_party/QNNPACK

#跨机器训练的通信库
third_party/gloo

#Intel开源的使用MKL-DNN做的神经网络加速库
third_party/ideep
```

## 参考
- PyTorch ATen代码的动态生成 https://zhuanlan.zhihu.com/p/55966063
- Pytorch1.3源码解析-第一篇 https://www.cnblogs.com/jeshy/p/11751253.html