# PyTorch引擎的主要模块及初始化

## 主要内容

本章对PyTorch的整体架构做了初步的分析，这部分也是理解PyTorch核心引擎工作机制的关键部分，在这里我们力图回答以下几个问题：
<ol>
<li>PyTorch从上层到C++的底层包括哪些重要的模块</li>
<li>这些模块是如何初始化的</li>
<li>从设计上看，这些模块是如何配合的</li>
</ol>

## PyTorch的核心模块

- PythonAPI
- C++部分Engine 
- THP
- ATen
- JITwdq

```bash

src
!--- ATen       # Tensor相关操作的C++接口
|--- TH         # Tensor的CPU实现
|--- THC        # Tensor的CUDA实现
|--- THCUNN     # 神经网络的CUDA实现
|--- THNN       # 神经网络的CPU实现

torch
|--- csrc       # Torch C++ 扩展模块的实现代码
      |--- module.cpp       # Torch C++ 扩展模块的初始化及入口代码

```

## PyTorch的C++扩展模块初始化
C++扩展模块_C可以说是PyTorch的核心，是PyTorch代码量最大最复杂的部分，下面我们来看看这个模块是如何加载及初始化的。

### C++扩展模块的加载
在加载torch模块的时候，python会执行torch/__init__.py. 其中会加载_C模块，根据Python3的规范，如果某个模块是C++实现的动态库，该库的名称应该为<Module>.cpython-<python version>-<arch>-<os>.so，在linux环境下，对应的就是_C.cpython-37m-x86_64-linux-gnu.so。

加载这个动态库后，会调用其中的initModule()函数。
在这个函数中，进行了一系列的初始化工作

```C++
PyObject* initModule() {

  // ...

  // 这里收集_C模块所需要的方法
  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());

  THPUtils_addPyMethodDefs(methods, THCPModule_methods());

  THPUtils_addPyMethodDefs(methods, torch::distributed::c10d::python_functions());

  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::python_functions());
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::distributed::rpc::testing::python_functions());

  // 下面开始创建_C模块
  static struct PyModuleDef torchmodule = {
     PyModuleDef_HEAD_INIT,
     "torch._C",
     nullptr,
     -1,
     methods.data()
  };
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  THPSize_init(module);
  THPDtype_init(module);
  THPDTypeInfo_init(module);
  THPLayout_init(module);
  THPMemoryFormat_init(module);
  THPQScheme_init(module);
  THPDevice_init(module);
  THPStream_init(module);

  // 初始化Tensor类型
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));
  // NOTE: We need to be able to access OperatorExportTypes from ONNX for use in
  // the export side of JIT, so this ONNX init needs to appear before the JIT
  // init.
  torch::onnx::initONNXBindings(module);
  torch::jit::initJITBindings(module);
  torch::monitor::initMonitorBindings(module);
  torch::impl::dispatch::initDispatchBindings(module);
  torch::throughput_benchmark::initThroughputBenchmarkBindings(module);
  torch::autograd::initReturnTypes(module);
  torch::autograd::initNNFunctions(module);
  torch::autograd::initFFTFunctions(module);
  torch::autograd::initLinalgFunctions(module);
  torch::autograd::initSparseFunctions(module);
  torch::autograd::initSpecialFunctions(module);
  torch::autograd::init_legacy_variable(module);
  torch::python::init_bindings(module);
  torch::lazy::initLazyBindings(module);
#ifdef USE_CUDA
  torch::cuda::initModule(module);
#endif
  ASSERT_TRUE(THPStorage_init(module));

#ifdef USE_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back into
  // C, so these lines have to execute first)..
  THCPStream_init(module);
  THCPEvent_init(module);
  THCPGraph_init(module);
#endif

  auto set_module_attr = [&](const char* name, PyObject* v, bool incref = true) {
    // PyModule_AddObject steals reference
    if (incref) {
      Py_INCREF(v);
    }
    return PyModule_AddObject(module, name, v) == 0;
  };

  // ...

  ASSERT_TRUE(set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

  // ...

  py::enum_<at::native::ConvBackend>(py_module, "_ConvBackend")
    .value("CudaDepthwise2d", at::native::ConvBackend::CudaDepthwise2d)
    .value("CudaDepthwise3d", at::native::ConvBackend::CudaDepthwise3d)
    .value("Cudnn", at::native::ConvBackend::Cudnn)
    .value("CudnnTranspose", at::native::ConvBackend::CudnnTranspose)
    .value("Empty", at::native::ConvBackend::Empty)
    .value("Miopen", at::native::ConvBackend::Miopen)
    .value("MiopenDepthwise", at::native::ConvBackend::MiopenDepthwise)
    .value("MiopenTranspose", at::native::ConvBackend::MiopenTranspose)
    .value("Mkldnn", at::native::ConvBackend::Mkldnn)
    .value("MkldnnEmpty", at::native::ConvBackend::MkldnnEmpty)
    .value("NnpackSpatial", at::native::ConvBackend::NnpackSpatial)
    .value("Overrideable", at::native::ConvBackend::Overrideable)
    .value("Slow2d", at::native::ConvBackend::Slow2d)
    .value("Slow3d", at::native::ConvBackend::Slow3d)
    .value("SlowDilated2d", at::native::ConvBackend::SlowDilated2d)
    .value("SlowDilated3d", at::native::ConvBackend::SlowDilated3d)
    .value("SlowTranspose2d", at::native::ConvBackend::SlowTranspose2d)
    .value("SlowTranspose3d", at::native::ConvBackend::SlowTranspose3d)
    .value("Winograd3x3Depthwise", at::native::ConvBackend::Winograd3x3Depthwise)
    .value("Xnnpack2d", at::native::ConvBackend::Xnnpack2d);

  py_module.def("_select_conv_backend", [](
        const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt,
        at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_,
        bool transposed_, at::IntArrayRef output_padding_, int64_t groups_) {
      return at::native::select_conv_backend(
          input, weight, bias_opt, stride_, padding_, dilation_, transposed_, output_padding_, groups_);
  });

  py::enum_<at::LinalgBackend>(py_module, "_LinalgBackend")
    .value("Default", at::LinalgBackend::Default)
    .value("Cusolver", at::LinalgBackend::Cusolver)
    .value("Magma", at::LinalgBackend::Magma);

  py_module.def("_set_linalg_preferred_backend", [](at::LinalgBackend b) {
    at::globalContext().setLinalgPreferredBackend(b);
  });
  py_module.def("_get_linalg_preferred_backend", []() {
    return at::globalContext().linalgPreferredBackend();
  });

  // ...  

  return module;
  END_HANDLE_TH_ERRORS
}
```


## 参考
- https://blog.csdn.net/Xixo0628/article/details/112603174
- https://blog.csdn.net/Xixo0628/article/details/112603174
- https://pytorch.org/blog/a-tour-of-pytorch-internals-1/#the-thptensor-type


# 第三章 PyTorch中重要的数据结构

## Tensor

在C++中，Tensor的定义在

## TensorOption

Note: 参考注释吧

TensorOption是设计用来构造Tensor的工具。

在C++中没有python中的keyword参数机制，比如这段代码：
```python
torch.zeros(2, 3, dtype=torch.int32)
```
在keyword参数机制下，参数的顺序和定义的可能不一样。因此在C++中实现这些函数时，将TensorOptions作为最后一个参数附在函数末尾，可以协助对参数的解析。

实际使用时，at::zeros()系列函数隐式的使用TensorOptions。 TensorOptions可以看作是一个字典。


```C++
// c10/core/TensorOptions.h



```

## Node
Node的定义在torch/csrc/autograd/function.h中。

从名称上不难看出，Node代表计算图中的节点。计算图除了节点之外，还会有边，也就是Edge.

Tensor中方法grad_fn()返回的就是一个Node

## Edge
Node的定义在torch/csrc/autograd/edge.h中。


## VariableHooks
获取Tensor的grad_fn()时，使用VariableHooks这个类来返回的，而且逻辑很复杂，还没看懂

https://blog.csdn.net/u012436149/article/details/69230136

这里要注意的是，hook 只能注册到 Module 上，即，仅仅是简单的 op 包装的 Module，而不是我们继承 Module时写的那个类，我们继承 Module写的类叫做 Container。
每次调用forward()计算输出的时候，这个hook就会被调用。它应该拥有以下签名：

可以看到，当我们执行model(x)的时候，底层干了以下几件事：

    调用 forward 方法计算结果

    判断有没有注册 forward_hook，有的话，就将 forward 的输入及结果作为hook的实参。然后让hook自己干一些不可告人的事情。

### register_backward_hook

在module上注册一个bachward hook。此方法目前只能用在Module上，不能用在Container上，当Module的forward函数中只有一个Function的时候，称为Module，如果Module包含其它Module，称之为Container

每次计算module的inputs的梯度的时候，这个hook会被调用。hook应该拥有下面的signature。

hook(module, grad_input, grad_output) -> Tensor or None

如果module有多个输入输出的话，那么grad_input grad_output将会是个tuple。
hook不应该修改它的arguments，但是它可以选择性的返回关于输入的梯度，这个返回的梯度在后续的计算中会替代grad_input。

这个函数返回一个 句柄(handle)。它有一个方法 handle.remove()，可以用这个方法将hook从module移除。

从上边描述来看，backward hook似乎可以帮助我们处理一下计算完的梯度。看下面nn.Module中register_backward_hook方法的实现，和register_forward_hook方法的实现几乎一样，都是用字典把注册的hook保存起来。


## Backward函数注册流程

```C++
initialize_autogenerated_functionsEverything();
   addClass<AddBackward0>(AddBackward0Class,"AddBackward0", AddBackward0_properties);
        _initFunctionPyTypeObject();
            
        registerCppFunction();
            cpp_function_types[idx] = type
```