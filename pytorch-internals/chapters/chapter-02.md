# PyTorch的源代码结构

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

## C10
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

## ATen
ATen，来自于 A TENsor library for C++11的缩写；PyTorch的C++ tensor library。ATen部分有大量的代码是来声明和定义Tensor运算相关的逻辑的，除此之外，PyTorch还使用了aten/src/ATen/gen.py来动态生成一些ATen相关的代码。ATen基于C10，Gemfield本文讨论的正是这部分；

## Caffe2
为了复用，2018年4月Facebook宣布将Caffe2的仓库合并到了PyTorch的仓库,从用户层面来复用包含了代码、CI、部署、使用、各种管理维护等。caffe2中network、operators等的实现，会生成libcaffe2.so、libcaffe2_gpu.so、caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so（caffe2 CPU Python 绑定）、caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so（caffe2 CUDA Python 绑定），基本上来自旧的caffe2项目)

## Torch

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

## 参考
- PyTorch ATen代码的动态生成 https://zhuanlan.zhihu.com/p/55966063
- Pytorch1.3源码解析-第一篇 https://www.cnblogs.com/jeshy/p/11751253.html