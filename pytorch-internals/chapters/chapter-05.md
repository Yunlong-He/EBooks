# 基于C++的算子实现

## 主要内容
- PyTorch中算子的实现方式
- 源代码的组织
- 运行代码分析
- 自定义算子的实现

## 一个简单的例子

我们先从一个简单的例子出发，看看PyTorch中Python和C++是怎样一起工作的。

```python
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
```

在_C模块初始化的时候，THPVariable这个类型绑定了相应的方法，可以在执行加法操作的时候，调用的是THPVariable_add()这个函数。


```C++
PyMethodDef variable_methods[] = {
  // These magic methods are all implemented on python object to wrap NotImplementedError
  {"__add__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__radd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add>), METH_VARARGS | METH_KEYWORDS, NULL},
  {"__iadd__", castPyCFunctionWithKeywords(TypeError_to_NotImplemented_<THPVariable_add_>), METH_VARARGS | METH_KEYWORDS, NULL},

  ...
}
```

THPVariable_add()方法的具体实现代码是生成的，因此我们在原始的模板文件中可以找到使用这个函数，真正的实现有多个，对应不同的调用方式。在这个例子里，对应调用的是下面这个实现：

```C++
// torch/csrc/autograd/generated/python_variable_methods.cpp [generated file]

static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, self_, args, kwargs, THPVariableClass, "torch.Tensor");
  }
  switch (_r.idx) {
    case 0: {
      // [deprecated] aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

      auto dispatch_add = [](const at::Tensor & self, const at::Scalar & alpha, const at::Tensor & other) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.scalar(0), _r.tensor(1)));
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor

      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```
其中 PythonArgParser 定义了这个函数的几类参数，并将Python调用的参数转换成对应的C++类型，在这个例子里，调用的参数符合第二组定义，因此_r.index为1，最后调用的是下面这个方法：



```C++
// aten/src/ATen/core/TensorBody.h

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
inline at::Tensor Tensor::add(const at::Tensor & other, const at::Scalar & alpha) const {
    return at::_ops::add_Tensor::call(const_cast<Tensor&>(*this), other, alpha);
}
```

```C++
// ./build/aten/src/ATen/Operators_2.cpp [generated file]

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, name, "aten::add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, schema_str, "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema> create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}
```

这里创建的op的类型是c10::OperatorHandle

## Dispatcher机制

所有的算子都是注册在Dispatcher里的，在调用的时候，根据函数名词和传递的参数类型，dispatcher会寻找相应的实现并进行调用；

```C++
class TORCH_API Dispatcher final {
private:

  struct OperatorDef final { ... };

public:
  static Dispatcher& realSingleton();

  C10_ALWAYS_INLINE static Dispatcher& singleton() { ...  }

  c10::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  c10::optional<OperatorHandle> findOp(const OperatorName& operator_name);

  const std::vector<OperatorName> getAllOpNames();

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return (Args...)>& op, Args... args) const;

  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const;

  // Invoke an operator via the boxed calling convention using an IValue stack
  void callBoxed(const OperatorHandle& op, Stack* stack) const;

  // TODO: This will only be useful if we write a backend fallback that plumbs dispatch keys (currently there are none)
  // See Note [Plumbing Keys Through The Dispatcher]
  void redispatchBoxed(const OperatorHandle& op, DispatchKeySet dispatchKeySet, Stack* stack) const;


  RegistrationHandleRAII registerDef(FunctionSchema schema, std::string debug);
  RegistrationHandleRAII registerImpl(OperatorName op_name, c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, c10::optional<impl::CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema, std::string debug);

  RegistrationHandleRAII registerName(OperatorName op_name);

  RegistrationHandleRAII registerFallback(DispatchKey dispatch_key, KernelFunction kernel, std::string debug);

  RegistrationHandleRAII registerLibrary(std::string ns, std::string debug);

  std::vector<OperatorName> getRegistrationsForDispatchKey(c10::optional<DispatchKey> k) const;

private:
  // ...

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  ska::flat_hash_map<std::string, std::string> libraries_;

  std::array<impl::AnnotatedKernel, num_runtime_entries> backendFallbackKernels_;

  // ...
};

```

这里看到两种注册的类型，一种是OperatorHandler，注册到operatorLookupTable_中，可以根据OperatorName查询，另一种是Function，一组Function注册到Library之后，再将Library注册到libraries_。

比如对于例子中的 y = x + 2这条语句，dispatcher会查询到一个OperatorHandler op
， op.operatorDef_->op.name_就是OperatorName("aten::add"，"Tensor")，但是注册的kernelfunction很多。


```C++
// ./aten/src/ATen/core/dispatch/Dispatcher.h

class TORCH_API OperatorHandle {
public:
  OperatorHandle(OperatorHandle&&) noexcept = default;
  // ...

  // See [Note: Argument forwarding in the dispatcher] for why Args doesn't use &&
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }

  // ...

private:
  // ...
  Dispatcher::OperatorDef* operatorDef_;
  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};
```
OperatorHandle的call()方法会调用Dispather::call()方法。

继续跟踪，会走到
```Bash
at::native::AVX2::cpu_kernel_vec<> (grain_size=32768, vop=..., op=..., iter=...)
    at ../aten/src/ATen/native/cpu/Loops.h:349


#0  at::native::AVX2::cpu_kernel_vec<> (grain_size=32768, vop=..., op=..., iter=...)
    at ../aten/src/ATen/native/cpu/Loops.h:349
#1  at::native::(anonymous namespace)::<lambda()>::operator() (__closure=<optimized out>)
    at /lab/tmp/pytorch/build/aten/src/ATen/UfuncCPUKernel_add.cpp:61
#2  at::native::(anonymous namespace)::add_kernel (iter=..., alpha=...)
    at /lab/tmp/pytorch/build/aten/src/ATen/UfuncCPUKernel_add.cpp:61
#3  0x00007fffe717e7be in at::(anonymous namespace)::wrapper_add_Tensor (self=..., other=..., alpha=...)
    at aten/src/ATen/RegisterCPU.cpp:1595


(gdb) bt
#0  at::native::AVX2::vectorized_loop<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>&, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)>&> (vop=..., op=..., S=2, n=4, data_=0x7fffffffd1c0)
    at ../aten/src/ATen/native/cpu/Loops.h:212
#1  at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)> >::<lambda(size_t)>::operator() (idx=2, __closure=<optimized out>)
    at ../aten/src/ATen/native/cpu/Loops.h:287
#2  at::native::AVX2::unroll_contiguous_scalar_checks<function_traits<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)> >, at::native::AVX2::VectorizedLoop2d<op_t, vop_t>::operator()(char**, const int64_t*, int64_t, int64_t) [with op_t = at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>; vop_t = at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)>]::<lambda(size_t)>, 1> (
    cb=..., strides=0x7fffffffd300) at ../aten/src/ATen/native/cpu/Loops.h:246
#3  at::native::AVX2::unroll_contiguous_scalar_checks<function_traits<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)> >, at::native::AVX2::VectorizedLoop2d<op_t, vop_t>::operator()(char**, const int64_t*, int64_t, int64_t) [with op_t = at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>; vop_t = at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)>]::<lambda(size_t)>, 0, 1> (
    cb=..., strides=0x7fffffffd300) at ../aten/src/ATen/native/cpu/Loops.h:248
#4  at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)> >::operator() (size1=1, size0=4, strides=0x7fffffffd300, base=0x0, this=0x7fffffffd4e0)
    at ../aten/src/ATen/native/cpu/Loops.h:283
#5  c10::function_ref<void(char**, long int const*, long int, long int)>::callback_fn<at::native::AVX2::VectorizedLoop2d<at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(scalar_t, scalar_t)>, at::native::(anonymous namespace)::add_kernel(at::TensorIteratorBase&, const c10::Scalar&)::<lambda()>::<lambda(at::vec::AVX2::Vectorized<float>, at::vec::AVX2::Vectorized<float>)> > >(intptr_t, char **, const long *, long, long) (callable=callable@entry=140737488344288, 
    params#0=params#0@entry=0x7fffffffd270, params#1=params#1@entry=0x7fffffffd300, params#2=params#2@entry=4, 
    params#3=params#3@entry=1) at ../c10/util/FunctionRef.h:43


```


## Dispatcher

Dispatcher的作用是根据实际的上下文选择不同的operator实现，

## 算子的注册过程
增加新的算子时，需要先使用TORCH_LIBRARY定义算子的schema，然后使用宏 TORCH_LIBRARY_IMPL来注册该算子在cpu、cuda、XLA等上的实现。注册的时候，需要指定namespace及该namespace下的dispatch_key，如果注册的是fallback实现（缺省实现）,namespace可以使用“_”。


下面我们看一下这两个宏的实现：

```C++
#define TORCH_LIBRARY(ns, m)                                                   \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);                        \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF,                                                     \
      &TORCH_LIBRARY_init_##ns,                                                \
      #ns,                                                                     \
      c10::nullopt,                                                            \
      __FILE__,                                                                \
      __LINE__);                                                               \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)


#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)

#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                             \
  static void C10_CONCATENATE(                                         \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);    \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(        \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(              \
      torch::Library::IMPL,                                            \
      c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check( \
          c10::DispatchKey::k)>(                                       \
          []() {                                                       \
            return &C10_CONCATENATE(                                   \
                TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid);           \
          },                                                           \
          []() { return [](torch::Library&) -> void {}; }),            \
      #ns,                                                             \
      c10::make_optional(c10::DispatchKey::k),                         \
      __FILE__,                                                        \
      __LINE__);                                                       \
  void C10_CONCATENATE(                                                \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

```

在VariableTypeEverything.cpp中，有这样一条语句：
```C++
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  ...
}
```
展开之后的形式如下：

```C++
  
static void TORCH_LIBRARY_IMPL_init_aten_Autograd_C10_UID(torch::Library&);
  static const torch::detail::TorchLibraryInit 
      TORCH_LIBRARY_IMPL_static_init_aten_Autograd_C10_UID(
      torch::Library::IMPL,
      c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check(
          c10::DispatchKey::k)>(
          []() {
            return & TORCH_LIBRARY_IMPL_init_aten_Autograd_C10_UID;
          },
          []() { return [](torch::Library&) -> void {}; }),
      #ns,                                                             \
      c10::make_optional(c10::DispatchKey::k),                         \
      __FILE__,                                                        \
      __LINE__);                                                       \
  void C10_CONCATENATE(                                                \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

```

对于每一个dispatch_key, 宏TORCH_LIBRARY_IMPL定义了一个函数，允许用户在这个函数体内注册


例如，在下面的代码中，注册了包括add_Tensor在内的多个算子。

```C++
// torch/csrc/autograd/generated/VariableTypeEveryThing.cpp

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  // ...
  m.impl("add.Tensor",
         TORCH_FN(VariableType::add_Tensor)
  );
  m.impl("add.Scalar",
         TORCH_FN(VariableType::add_Scalar)
  );
  // ...
}
```



```C++
THPVariable_add ->


```

## 自定义算子的实现过程

### 原生算子的实现
所谓“原生”，指的就是内置在PyTorch中的算子，跟随PyTorch一起编译生成，可以同"torch.xxx"等方式使用的算子。

由于原生算子的数量非常多，处于效率和可用性的考虑，在不同的平台上可能会有实现，另外算子要支持注册到torch模块、自动微分、jit等，也造成了算子之间会有大量重复的操作，在PyTorch中，用模板的方式隐藏了这些细节，当然也带来了一定的复杂性。

很多原生算子的模板定义在native_functions.yaml中，比如sigmoid函数：

```
# aten/src/ATen/native/native_functions.yaml

- func: sigmoid(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: sigmoid.out
  variants: function, method
  dispatch:
    QuantizedCPU: sigmoid_quantized_cpu
    MkldnnCPU: mkldnn_sigmoid


- func: sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
  python_module: nn
  structured_delegate: sigmoid_backward.grad_input

```
其中：
- func字段定义了算子的名称和输入输出参数。
- device_check: 暂时还不清楚用途，在模板里都是NoCheck。
- structured_delegate: sigmoid.out
- variants字段生命这个算子的类型和使用方式，function表明sigmoid这个算子可以通过函数torch.sigmoid()中，method则表明可以像使用对象的方法一样使用，比如tensorA是一个张量，可以用tensorA.sigmoid()来调用。
- dispatch字段定义了在不同的平台或者优化方式下该算子的变体。这里针对使用量化方式运行时，会调用相应的量化实现sigmoid_quantized_cpu，需要使用mkldnn加速的时候，调用mkldnn_sigmoid。
- python-module字段定义了该算法会被注册到的Python模块。

sigmoid函数是机器学习中最基本的函数之一，其公式如下：
$$Sigmoid(x) = \frac{1}{1+e^{-x}}$$

我们在使用sigmoid函数时，调用的是torch.nn.Sigmoid函数，其背后则是调用了torch.sigmoid()函数，也就是上面定义的native实现。

```Python
class Sigmoid(Module):
    r"""Applies the element-wise function:
    Examples::
        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input)
```


在tools/autograd/derivatives.yaml中，定义了算子的前向计算输出反向计算梯度的对应关系，比如sigmoid算子的对应关系如下：

```
- name: sigmoid(Tensor self) -> Tensor
  self: sigmoid_backward(grad, result)
  result: auto_element_wise
```

在native_functions.yaml中只是声明了sigmoid算子，具体的算子实现是和平台相关的，因此要到各个平台目录下去寻找，例如cpu下的sigmoid算子实现在cpu/UnaryOpsKernel.cpp里：

```C++
// aten/src/ATen/native/cpu/UnaryOpsKernel.cpp

static void sigmoid_kernel(TensorIteratorBase& iter) {
  if (iter.common_dtype() == kBFloat16) {
    cpu_kernel_vec(
        iter,
        [=](BFloat16 a) -> BFloat16 {
          float a0 = static_cast<float>(a);
          return static_cast<float>(1) / (static_cast<float>(1) + std::exp((-a0)));
        },
        [=](Vectorized<BFloat16> a) {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
          a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "sigmoid_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });
    });
  }
}

REGISTER_DISPATCH(sigmoid_stub, &CPU_CAPABILITY::sigmoid_kernel);


// aten/src/ATen/native/cpu/BinaryOpsKernel.cpp

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.dtype())) {
    // ......
  } else if (iter.dtype() == kBFloat16) {
    // ......
  } else {
    // ......
  }
}

```



```C++
// aten/src/ATen/native/cpu/UnaryOps.cpp

CREATE_UNARY_FLOAT_META_FUNC(sigmoid)

CREATE_UNARY_TORCH_IMPL_FUNC(sigmoid_out, sigmoid_stub)
DEFINE_DISPATCH(sigmoid_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)


```
在sigmoid_kernel()的实现里，根据传输Tensor类型的不同，构建了不同的匿名函数，然后调用cpu_kernel_vec()对不同的Tensor分别进行计算。可以认为这些匿名函数是真正的sigmoid的实现，cpu_kernel_vec()是一个通用函数，

sigmoid_kernel是sigmoid算子在cpu下的实现，当然即使在CPU下，sigmoid函数也有多种形式，除了普通的浮点计算外，也要支持半浮点的形式。
AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES宏有三个参数：
- iter.common_dtype()，指明操作的Tensor属于哪种类型。
- "sigmoid_cpu"， 算子的名称
- 匿名函数，调用了cpu_kernel_vec

在aten/src/ATen/native/cpu/Loops.cpp中，有两个cpu_kernel相关的函数，由于cpu下的源文件在编译的时候会加上AVX/AVX2的选项，因此能够充分利用CPU中的现代化指令，不过这两个函数向量化的实现方式不太一样。
- cpu_kernel()：依赖于编译器自动实现计算的向量化
- cpu_kernel_vec()：使用x86 SIMD原语实现向量化。
一般来讲，使用cpu_kernel_vec()的时候，说明实现该算子的实现是经过精心优化的，效率会高一些。

例如用这两个函数实现浮点数相乘的算子，可以这样实现：

```C++

cpu_kernel(iter, [](float a, float b) { return a * b; });

cpu_kernel_vec(iter,
     [](float a, float b) { return a * b; },
     [](Vectorized<float> a, Vectorized<float> b) { return a * b; });
```

下面我们看一下cpu_kernel_vec()函数的实现：

```C++
// aten/src/ATen/native/cpu/Loops.cpp

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  c10::guts::if_constexpr<check_dynamic_cast>([&] {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  });

  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  iter.cast_outputs();
}

```
可以看到，对每个Tensor，又调用了make_vectorized_loop2d()
```C++
// aten/src/ATen/native/cpu/Loops.cpp
template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(
    const op_t &op, const vop_t &vop) {
  return VectorizedLoop2d<op_t, vop_t>(op, vop);
}

template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;
  vop_t vop;

  using traits = function_traits<op_t>;
  static constexpr int ntensors = traits::arity + 1;
  using data_t = std::array<char*, ntensors>;

  VectorizedLoop2d(const op_t &op, const vop_t &vop):
    op(op), vop(vop) {}

  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, ntensors, data.data());
    const int64_t *outer_strides = &strides[ntensors];

    if (is_contiguous<traits>(strides)) {
      for (const auto i : c10::irange(size1)) {
        (void)i;
        vectorized_loop(data.data(), size0, 0, op, vop);
        advance(data, outer_strides);
      }
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            basic_loop(data.data(), strides, 0, size0, op);
            advance(data, outer_strides);
          }
        }
      });
    }
  }
};
```
很明显，VectorizedLoop2d的主要工作就是根据Tensor的stride的不同，选择不同的调用模式，但最终不管是调用vectorized_loop()还是调用basic_loop()，都是对一块连续的Tensor进行计算。可以这样计算的。

现在我们回到当初sigmoid函数的实现部分，其中对每个Tensor的操作函数实现是这样的：

```C++
// aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });

// aten/src/ATen/native/cpu/vec/vec256/vec256_float.h
 Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf8_u10(values));
  }
```

在代码中可以看出，对应cpu的实现有很多，实际运行时会根据不同的平台和数据类型调用相应的实现，以达到比较好的性能。而由于CPU上的向量计算有很多库可以用，这里用到的是sleef库的实现。

> https://blog.csdn.net/yelede2009/article/details/120411361
> 有各种函数库以向量方式来计算数学函数，例如：对数、幂函数、三角函数等。这些函数库对向量化数学代码是有用的。
> 有两种不同种类的向量数学库：长向量库和短向量库。来看看它们的不同。假设要计算1000个数字的某个函数。用长向量库，把1000个数字的数组作为参数传给库函数，这
> 个库函数存储这1000个结果到另一个数组。使用长向量版库函数的缺点是，如果要做一系列计算，在下一次调用函数前就需要存储中间结果到一个临时数组中。用短向量版本
> 的向量库，可以把数据集拆分为子向量来适配向量寄存器。如果向量寄存器可以处理4个数字，那么需要调用250此库函数。这个库函数会返回结果到向量寄存器中，可以直接
> 被下一次计算利用，而不需要存储中间结果到RAM中。这可能更快。然而，短向量的库函数可能是不利的，如果计算序列形成了长依赖链。
>这是一些长向量函数库：
>
>    Intel 向量数学库（VML, MKL）。工作在x86平台。这些库函数在非Intel的CPU上会低效，除非重写了Intel cpu分发器。
>    Intel的IPP。工作在x86平台。也适用于非Intel的CPU。包含很多统计、信号处理和图像处理函数。
>    Yeppp。开源库。支持x86和ARM平台，多种编程语言。参考Yeppp。
>
>这是一些短向量库：
>
>    Sleef库。支持多种平台。开源。参考www.sleef.org。
>    Intel短向量库（SVML）。Intel编译器提供，被自动向量化调用。Gnu编译器可以通过选项-mveclibabi=svml使用这个库。如果用的是非Intel的CPU，也可以使用。
>    AMD LIBM库。只支持64位Linux平台。没有FMA4指令集时，性能会降低。Gnu通过-mveclibabi=acml选项使用。
>    VCL库。个人开发。参考https://github.com/vectorclass。

Dispatch的过程似乎有些复杂，有很多宏处理，更是导致不容易看懂。
```C++
// aten/src/ATen/Dispatch.h

#define AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                     \
  AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, scalar_t, __VA_ARGS__)


#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                   \
  [&] {                                                                        \
    const auto& the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                   \
    switch (_st) {                                                             \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__)  \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");         \
    }                                                                          \
  }()
  ```

宏AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES

## 参考
- https://pytorch.org/tutorials/advanced/dispatcher.html
- http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
- https://blog.csdn.net/Chris_zhangrx/article/details/119512418
- https://zhuanlan.zhihu.com/p/67834038
- https://blog.csdn.net/xixiaoyaoww/article/details/112211025
- pytorch中的dispatcher https://zhuanlan.zhihu.com/p/390049109
- [Pytorch 源码阅读] —— 谈谈 dispatcher（二）https://blog.csdn.net/Chris_zhangrx/article/details/119512418
- [Pytorch 源码阅读] —— 谈谈 dispatcher（一） https://blog.csdn.net/Chris_zhangrx/article/details/119489853

- https://zhuanlan.zhihu.com/p/349560723
- https://zhuanlan.zhihu.com/p/499979372
- 这可能是关于Pytorch底层算子扩展最详细的总结了 https://wenku.baidu.com/view/1415b43ac181e53a580216fc700abb68a982ad3d.html






