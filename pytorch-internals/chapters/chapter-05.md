# 基于C++的算子实现

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
//torch/csrc/autograd/generated/python_variable_methods.cpp [generated file]

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

## 参考
- https://pytorch.org/tutorials/advanced/dispatcher.html
- http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/
- https://blog.csdn.net/Chris_zhangrx/article/details/119512418








