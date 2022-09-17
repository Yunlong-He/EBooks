
# 第四章 PyTorch的编译

## 初步介绍

大多数情况下我们只需要安装PyTorch的二进制版本即可，即可进行普通的模型开发训练了，但如果要深入了解PyTorch的实现原理，或者对PyTorch做一些优化改进，需要从PyTorch的源码开始进行编译安装，在PyTorch的官网里有从源码安装的说明。

以下的编译过程是基于

## 编译环境准备

根据官方文档，建议安葬Python 3.7或以上的环境，而且需要C++14的编译器，比如clang，一开始我在ubuntu中装了clang，是6.0，结果出现了一些编译选项的错误，于是卸载clang，安装gcc后，c++的版本是7.5。

Python的环境我也根据建议安装了Anaconda.

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

如果没有什么问题，编译的最后输出如下：

```bash
......

building 'torch._C' extension
creating build/temp.linux-x86_64-3.7
creating build/temp.linux-x86_64-3.7/torch
creating build/temp.linux-x86_64-3.7/torch/csrc
gcc -pthread -B /root/anaconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/anaconda3/include/python3.7m -c torch/csrc/stub.c -o build/temp.linux-x86_64-3.7/torch/csrc/stub.o -Wall -Wextra -Wno-strict-overflow -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-deprecated-declarations -fno-strict-aliasing -Wno-missing-braces
gcc -pthread -shared -B /root/anaconda3/compiler_compat -L/root/anaconda3/lib -Wl,-rpath=/root/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/stub.o -L/lab/tmp/pytorch/torch/lib -ltorch_python -o build/lib.linux-x86_64-3.7/torch/_C.cpython-37m-x86_64-linux-gnu.so -Wl,-rpath,$ORIGIN/lib
building 'torch._C_flatbuffer' extension
gcc -pthread -B /root/anaconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/anaconda3/include/python3.7m -c torch/csrc/stub_with_flatbuffer.c -o build/temp.linux-x86_64-3.7/torch/csrc/stub_with_flatbuffer.o -Wall -Wextra -Wno-strict-overflow -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-deprecated-declarations -fno-strict-aliasing -Wno-missing-braces
gcc -pthread -shared -B /root/anaconda3/compiler_compat -L/root/anaconda3/lib -Wl,-rpath=/root/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/stub_with_flatbuffer.o -L/lab/tmp/pytorch/torch/lib -ltorch_python -o build/lib.linux-x86_64-3.7/torch/_C_flatbuffer.cpython-37m-x86_64-linux-gnu.so -Wl,-rpath,$ORIGIN/lib
building 'torch._dl' extension
gcc -pthread -B /root/anaconda3/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/anaconda3/include/python3.7m -c torch/csrc/dl.c -o build/temp.linux-x86_64-3.7/torch/csrc/dl.o
gcc -pthread -shared -B /root/anaconda3/compiler_compat -L/root/anaconda3/lib -Wl,-rpath=/root/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.7/torch/csrc/dl.o -o build/lib.linux-x86_64-3.7/torch/_dl.cpython-37m-x86_64-linux-gnu.so
-------------------------------------------------------------------------
|                                                                       |
|    It is no longer necessary to use the 'build' or 'rebuild' targets  |
|                                                                       |
|    To install:                                                        |
|      $ python setup.py install                                        |
|    To develop locally:                                                |
|      $ python setup.py develop                                        |
|    To force cmake to re-generate native build files (off by default): |
|      $ python setup.py develop --cmake                                |
|                                                                       |
-------------------------------------------------------------------------
```

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
> python setup.py clean
> python setup.py build
```

- lib库找不到
错误详情：No rule to make target '/usr/lib/x86_64-linux-gnu/libXXX.so
```bash
> find / -name "librt.so.*"
> ln -s /lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so

```
- c++命令找不到
```bash
> apt install g++
```
注意，如果安装clang，也可以编译，但c++的版本如果比较低，比如6.0，就容易出现C++ 命令编译开关没找到
的问题。

- 在PC上编译时Hang住

一般来说为了加快编译速度，编译大型项目时都会采用并行编译的方式，pytorch的编译也是，启动编译后，可以在另一个窗口使用top查看CPU占用情况。由于PC的核数比较少，当并行度比较高的时候，就容易造成死锁。

简单起见，在启动编译前，可以设置环境变量CMAKE_BUILD_PARALLEL_LEVEL来减少编译的并行度。

-- 编译Debug版本时出现internal compiler error

如果只是在编译Debug版本时出现，可能是和优化编译选项有冲突，因为优化编译选项-O1 -O2 -O3可能会重新排列代码导致代码对应出现问题，排查真正的问题非常困难，建议简单处理，对出现问题的编译部分去掉-g选项或者-O 选项。

PyTorch的编译由setup.py发起，但真正执行编译时，相关的命令写在build/build.ninja里，只要在这个文件里修改相关的编译参数，再重新启动编译即可。

## 参考


<ol>
<li>https://zhuanlan.zhihu.com/p/321449610</li>
<li> https://blog.51cto.com/SpaceVision/5072093</li>
<li> https://zhuanlan.zhihu.com/p/55204134</li>
<li> https://github.com/pytorch/pytorch#from-source </li>
<li> 从零开始编译PyTorch软件包 https://zhuanlan.zhihu.com/p/347084475 </li>
<li> Pytorch setup.py 详解 https://blog.csdn.net/Sky_FULLl/article/details/125652654</li>
<li> PyTorch 动态代码生成 https://zhuanlan.zhihu.com/p/55966063</li>
<li> PyTorch 动态代码生成 https://zhuanlan.zhihu.com/p/59425970</li>
</ol>
