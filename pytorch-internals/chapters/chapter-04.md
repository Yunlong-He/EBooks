
# 第四章 PyTorch的编译

## 初步介绍

大多数情况下我们只需要安装PyTorch的二进制版本即可，即可进行普通的模型开发训练了，但如果要深入了解PyTorch的实现原理，或者对PyTorch做一些优化改进，需要从PyTorch的源码开始进行编译安装，在PyTorch的官网里有从源码安装的说明。

以下的编译过程是基于

## 编译环境准备

编译PyTorch需要以下
    Python 3.7 or later (for Linux, Python 3.7.6+ or 3.8.1+ is needed)
    A C++14 compatible compiler, such as clang

We highly recommend installing an Anaconda environment. You will get a high-quality BLAS library (MKL) and you get controlled dependency versions regardless of your Linux distro.

If you want to compile with CUDA support, install the following (note that CUDA is not supported on macOS)

    NVIDIA CUDA 10.2 or above
    NVIDIA cuDNN v7 or above
    Compiler compatible with CUDA

Note: You could refer to the cuDNN Support Matrix for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware

If you want to disable CUDA support, export the environment variable USE_CUDA=0. Other potentially useful environment variables may be found in setup.py.

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are available here

If you want to compile with ROCm support, install

    AMD ROCm 4.0 and above installation
    ROCm is currently supported only for Linux systems.

If you want to disable ROCm support, export the environment variable USE_ROCM=0. Other potentially useful environment variables may be found in setup.py.

## PyTorch的setup.py

参考 https://blog.csdn.net/Sky_FULLl/article/details/125652654


## PyTorch 动态代码生成

参考 https://zhuanlan.zhihu.com/p/59425970
参考 https://zhuanlan.zhihu.com/p/55966063

PyTorch代码主要包括三部分：
- <b>C10</b>. C10是Caffe Tensor Library的缩写。PyTorch目前正在将代码从ATen/core目录下迁移到C10中，目前存放的都是最核心、精简的、基础的Tensor函数和接口。
- <b>ATen</b>，ATen是A TENsor library for C++11的缩写，是PyTorch的C++ tensor library。ATen部分有大量的代码是来声明和定义Tensor运算相关的逻辑的，除此之外，PyTorch还使用了aten/src/ATen/gen.py来动态生成一些ATen相关的代码。ATen基于C10。
- <b>Torch</b>，部分代码仍然在使用以前的快要进入历史博物馆的Torch开源项目，比如具有下面这些文件名格式的文件：
```text
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

C10目前最具代表性的一个class就是TensorImpl了，它实现了Tensor的最基础框架。继承者和使用者有：

## 编译步骤

### 编译第三方的库

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
## 代码生成

ATen的native函数是PyTorch目前主推的operator机制，作为对比，老旧的TH/THC函数（使用cwrap定义）将逐渐被ATen的native替代。ATen的native函数声明在native_functions.yaml文件中，然后实现在ATen/native目录下。移植AdaptiveMaxPooling2d op需要修改这个yaml文件：

## 常见问题

- submodule没有下载完整
  一个简单的处理办法是删除third_party下的相关目录，然后手动git clone即可。相关的git url定义在.submodule以及.gi/config中
- 编译时出现RPATH相关的问题
  处理办法是先运行clean命令，然后再编译

```bash
    # python setup.py clean
    # python setup.py build
```


## 参考

https://zhuanlan.zhihu.com/p/321449610

<ol>
<li> https://zhuanlan.zhihu.com/p/55204134</li>
<li> https://github.com/pytorch/pytorch#from-source </li>
<li> 从零开始编译PyTorch软件包 https://zhuanlan.zhihu.com/p/347084475 </li>
<li> Pytorch setup.py 详解 https://blog.csdn.net/Sky_FULLl/article/details/125652654</li>
<li> PyTorch 动态代码生成 https://zhuanlan.zhihu.com/p/55966063</li>
<li> PyTorch 动态代码生成 https://zhuanlan.zhihu.com/p/59425970</li>
</ol>
