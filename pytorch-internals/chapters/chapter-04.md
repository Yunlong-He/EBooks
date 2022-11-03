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
// torch/csrc/Module.cpp

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

## Torch 函数库的初始化

在Python层面，模块torch提供了非常多的函数，比如torch.abs()，torch.randn()， torch.ones()等等，在初始化_C模块的时候，这些函数也被注册到_C模块中。

```C++
// torch/csrc/autograd/python_variable.cpp

bool THPVariable_initModule(PyObject *module)
{
  // ...
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  // ...
  return true;
}
```
在下面的代码中，我们可以看到，相关的函数被收集到torch_functions中，同时这个函数列表也被注册到_C的_VariableFunctions这个子模块中。
```C++
// torch/csrc/autograd/python_torch_functions_manual.cpp

void initTorchFunctions(PyObject *module) {
  static std::vector<PyMethodDef> torch_functions;
  gatherTorchFunctions(torch_functions);
  THPVariableFunctions.tp_methods = torch_functions.data();
  
  //...
  if (PyModule_AddObject(module, "_VariableFunctionsClass",
                         reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }
  // PyType_GenericNew returns a new reference
  THPVariableFunctionsModule = PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject steals a reference
  if (PyModule_AddObject(module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}
```

在torch模块的初始化过程中，_C模块的子模块_VariableFunctions中的所有属性都被注册到torch模块中，当然也包括所有的函数。
```Python
# torch/__init__.py

for name in dir(_C._VariableFunctions):
    if name.startswith('__') or name in PRIVATE_OPS:
        continue
    obj = getattr(_C._VariableFunctions, name)
    obj.__module__ = 'torch'
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)
```

下面我们看看具体有哪些函数被注册了。函数列表是通过gatherTorchFunctions()来收集的，这个函数又调用了gatherTorchFunctions_0(), gatherTorchFunctions_1(), gatherTorchFunctions_2()这几个函数。

```C++
// torch/csrc/autograd/python_torch_functions_manual.cpp

void gatherTorchFunctions(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  torch_functions.assign(torch_functions_manual,
                         torch_functions_manual + num_functions);
  // NOTE: Must be synced with num_shards in tools/autograd/gen_python_functions.py
  gatherTorchFunctions_0(torch_functions);
  gatherTorchFunctions_1(torch_functions);
  gatherTorchFunctions_2(torch_functions);

  //...

```
为什么这样设计呢？大概有两个原因：
- 函数的数量很多，而且在不断的增加，需要方便扩展
- 函数大多是算子，算子和平台相关，每个算子有多种实现，同样为了在不同平台迁移扩展，PyTorch设计了代码生成机制来屏蔽通用的、繁琐的功能，这个生成机制也需要解耦。

gatherTorchFunctions_N()这几个函数是通过模板生成的，完成编译后，可以在下面的文件中找到：

```C++
// torch/csrc/autograd/generated/python_torch_functions_0.cpp

static PyMethodDef torch_functions_shard[] = {
  {"_cast_Byte", castPyCFunctionWithKeywords(THPVariable__cast_Byte), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  //...
  {"eye", castPyCFunctionWithKeywords(THPVariable_eye), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  {"rand", castPyCFunctionWithKeywords(THPVariable_rand), METH_VARARGS | METH_KEYWORDS | METH_STATIC, NULL},
  //...
};

void gatherTorchFunctions_0(std::vector<PyMethodDef> &torch_functions) {
  constexpr size_t num_functions = sizeof(torch_functions_shard) / sizeof(torch_functions_shard[0]);
  torch_functions.insert(
    torch_functions.end(),
    torch_functions_shard,
    torch_functions_shard + num_functions);
}


```

## Tensor

在Pytorch的早期版本中，Tensor被定义在TH模块中的THTensor类中，后来TH模块被移除了，也就有了更直观的Tensor类。

当前Tensor的定义在TensorBody.h中，

```C++
// torch/include/ATen/core/TensorBody.h

class TORCH_API Tensor: public TensorBase {
 public:
  Tensor(const Tensor &tensor) = default;
  Tensor(Tensor &&tensor) = default;

  using TensorBase::size;
  using TensorBase::stride;

  Tensor cpu() const {
    return to(options().device(DeviceType::CPU), /*non_blocking*/ false, /*copy*/ false);
  }

  // TODO: The Python version also accepts arguments
  Tensor cuda() const {
    return to(options().device(DeviceType::CUDA), /*non_blocking*/ false, /*copy*/ false);
  }

  void backward(const Tensor & gradient={}, ...) const {
    ...
  }
}
```

我们还可以看到，Tensor类本身的实现很少，大部分功能来自于其父类TensorBase。根据文档注释我们可以了解到，这样做的初衷是为了避免修改算子签名的时候造成太多模块的重新编译，因为Tensor是一个核心数据结构，几乎所有的模块都会依赖于Tensor。

```C++
// torch/include/ATen/core/TensorBase.h

class TORCH_API TensorBase {

  int64_t dim() const {
    return impl_->dim();
  }
  int64_t storage_offset() const {
    return impl_->storage_offset();
  }

  // ...

  bool requires_grad() const {
    return impl_->requires_grad();
  }
  bool is_leaf() const;
  TensorBase data() const;

  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
}
```

> https://blog.csdn.net/Chris_zhangrx/article/details/119086815
> c10::intrusive_ptr是PyTorch的内部智能指针实现，其工作方式如下：
> 首先完美转发所有的参数来构建 intrusive_ptr
> 用这些参数 new 一个新的 TTarget 类型的对象
> 用新的 TTarget 对象构造一个 intrusive_ptr
> 构造 intrusive_ptr 的同时对 refcount_ 和 weakcount_ 都加 1，
> 如果是默认构造，则两个引用计数都默认为 0，根据这个可以将通过 make_intrusive 构造的指针与堆栈上的会被自动析构的情况分开, 用来确保内存是我们自己分配的。

以后有机会我们再研究一下intrusive_ptr的实现，在此之前，我们主要关注impl_这个成员变量，也就是TensorImpl这个类的实现。

```C++
// c10/core/TensorImpl.h

struct C10_API TensorImpl : public c10::intrusive_ptr_target {

TensorImpl(
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type);

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  DispatchKeySet key_set() const {
    return key_set_;
  }

  int64_t dim() const {
    //...
  }
  bool is_contiguous(
    //...
  } 

  Storage storage_;

private:
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

 protected:
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;

  c10::VariableVersion version_counter_;

  std::atomic<impl::PyInterpreter*> pyobj_interpreter_;
  PyObject* pyobj_;

  c10::impl::SizesAndStrides sizes_and_strides_;

  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;

  caffe2::TypeMeta data_type_;
  c10::optional<c10::Device> device_opt_;

  bool is_contiguous_ : 1;

  bool storage_access_should_throw_ : 1;

  bool is_channels_last_ : 1;
  bool is_channels_last_contiguous_ : 1;
  bool is_channels_last_3d_ : 1;

  bool is_channels_last_3d_contiguous_ : 1;

  bool is_non_overlapping_and_dense_ : 1;

  bool is_wrapped_number_ : 1;

  bool allow_tensor_metadata_change_ : 1;

  bool reserved_ : 1;
  uint8_t sizes_strides_policy_ : 2;

  DispatchKeySet key_set_;

}

```

对于TensorImpl类来说，比较重要的成员变量有以下几个：
- storage_。这个变量存储了真正的张量数据
- autograd_meta_。存储反向传播所需要的元信息，如梯度计算函数和梯度等。
- pyobj_。Tensor所对应的Python Object
- data_type_。Tensor内的数据类型。
- device_opt_。存放Tensor的设备。
- 

下面我们看一下Tensor的存储，因为Tensor的存储方式和算子的计算息息相关，对性能的影响也非常的关键。
```C++
// c10/core/Storage.h

struct C10_API Storage {
  //...

 protected:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
}

```
和Tensor的定义类似，Storage也是使用StorageImpl类来隐藏其复杂的实现。因此我们主要关注StorageImpl的实现。

```C++
// c10/core/StorageImpl.h

struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  void* data() {
    return data_ptr_.get();
  }
  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

private:
  DataPtr data_ptr_;
  size_t size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;
  Allocator* allocator_;
}
```

StorageImpl的关键成员是data_ptr_, 其定义在这里：

```C++
// c10/core/Allocator.h

class C10_API DataPtr {
 private:
  c10::detail::UniqueVoidPtr ptr_;
  Device device_;

  
}

// c10/util/UniqueVoidPtr.h

class UniqueVoidPtr {
 private:
  // Lifetime tied to ctx_
  void* data_;
  std::unique_ptr<void, DeleterFnPtr> ctx_;

  // ...
}
```

现在我们知道，在C++的层面，张量被Tensor类型所表示，但是我们平时是使用Python语言来训练推理模型的，使用的自然是Python中的Tensor类型，那么在PyTorch中是如何实现从Python语言中的Tensor对象到C++中的Tensor对象的转换呢？

详细的过程我们留到后面的章节解释，不过机制并不复杂，PyTorch使用了THPVariable这个类型作为过渡，Python中的Tensor类

在前面初始化_C模块的时候，调用了THPVariable_initModule()这个函数，将Python中_TensorBase这个类型映射到THPVariableType这个C++类型上，

```C++
// torch/csrc/autograd/python_variable.cpp

bool THPVariable_initModule(PyObject *module)
{
  // ...
  PyModule_AddObject(module, "_TensorBase",   (PyObject *)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  // ...
  return true;
}

PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C._TensorBase", /* tp_name */
    // ...
    THPVariable_pynew, /* tp_new */
};

PyObject *THPVariable_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  TORCH_CHECK(type != &THPVariableType, "Cannot directly construct _TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);
  // WARNING: tensor is NOT guaranteed to be a fresh tensor; e.g., if it was
  // given a raw pointer that will refcount bump
  return THPVariable_NewWithVar(
      type,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status) {
 
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*) obj;
    // TODO: named constructor to avoid default initialization
    new (&v->cdata) MaybeOwned<Variable>();
    v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
    const auto& var = THPVariable_Unpack(v);
    var.unsafeGetTensorImpl()->init_pyobj(self_interpreter.get(), obj, status);
    if (check_has_torch_dispatch(obj)) {
      var.unsafeGetTensorImpl()->set_python_dispatch(true);
    }
  }
  return obj;
}

// torch/csrc/autograd/python_variable.h
struct THPVariable {
  PyObject_HEAD;
  c10::MaybeOwned<at::Tensor> cdata;
  PyObject* backward_hooks = nullptr;
};
```



## TensorOption

Note: 参考注释吧

TensorOption是设计用来构造Tensor的工具。

在C++中没有python中的keyword参数机制，比如这段代码：
```python
torch.zeros(2, 3, dtype=torch.int32)
```
在keyword参数机制下，参数的顺序和定义的可能不一样。因此在C++中实现这些函数时，将TensorOptions作为最后一个参数附在函数末尾，可以协助对参数的解析。

实际使用时，at::zeros()系列函数隐式的使用TensorOptions。 TensorOptions可以看作是一个字典。


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



## 参考
- https://blog.csdn.net/Xixo0628/article/details/112603174
- https://blog.csdn.net/Xixo0628/article/details/112603174
- https://pytorch.org/blog/a-tour-of-pytorch-internals-1/#the-thptensor-type
- PyTorch源码浅析(1)：THTensor https://blog.csdn.net/Xixo0628/article/details/112603174
- PyTorch源码浅析(1)：THTensor https://www.52coding.com.cn/2019/05/05/PyTorch1/

