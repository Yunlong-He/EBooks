
# 自动微分

## 数据结构

## TensorImpl是Tensor的实现
at::Tensor：shared ptr 指向 TensorImpl

TensorImpl：对 at::Tensor 的实现

    包含一个类型为 [AutogradMetaInterface](c10::AutogradMetaInterface) 的autograd_meta_，在tensor是需要求导的variable时，会被实例化为 [AutogradMeta](c10::AutogradMetaInterface) ，里面包含了autograd需要的信息

Variable: 就是Tensor，为了向前兼容保留的

    using Variable = at::Tensor;

    概念上有区别, Variable 是需要计算gradient的, Tensor 是不需要计算gradient的

    Variable的 AutogradMeta是对 [AutogradMetaInterface](c10::AutogradMetaInterface)的实现，里面包含了一个 Variable，就是该variable的gradient

    带有version和view

    会实例化 AutogradMeta , autograd需要的关键信息都在这里

```C++
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  // ...
public:
  Storage storage_;

private:
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

 protected:
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;

  c10::VariableVersion version_counter_;

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

### AutoGradMeta

    AutoGradMeta : 记录 Variable 的autograd历史信息

    包含一个叫grad_的 Variable， 即 AutoGradMeta 对应的var的梯度tensor

    包含类型为 Node 指针的 grad_fn （var在graph内部时）和 grad_accumulator（var时叶子时）, 记录生成grad_的方法

    包含 output_nr ，标识var对应 grad_fn的输入编号

    构造函数包含一个类型为 Edge的gradient_edge, gradient_edge.function 就是 grad_fn, 另外 gradient_edge.input_nr 记录着对应 grad_fn的输入编号，会赋值给 AutoGradMeta 的 output_nr

### Edge
autograd::Edge: 指向autograd::Node的一个输入

    包含类型为 Node 指针，表示edge指向的Node

    包含 input_nr， 表示edge指向的Node的输入编号

### Node

autograd::Node: 对应AutoGrad Graph中的Op

    是所有autograd op的抽象基类，子类重载apply方法

        next_edges_记录出边

        input_metadata_记录输入的tensor的metadata

    实现的子类一般是可求导的函数和他们的梯度计算op

    Node in AutoGrad Graph

        Variable通过Edge关联Node的输入和输出

        多个Edge指向同一个Var时，默认做累加

    call operator

        最重要的方法，实现计算

    next_edge

        缝合Node的操作

        获取Node的出边，next_edge(index)/next_edges()

        add_next_edge()，创建
## 前向计算

PyTorch通过tracing只生成了后向AutoGrad Graph.

代码是生成的，需要编译才能看到对应的生成结果

    gen_variable_type.py生成可导版本的op

    生成的代码在 pytorch/torch/csrc/autograd/generated/

    前向计算时，进行了tracing，记录了后向计算图构建需要的信息

    这里以relu为例，代码在pytorch/torch/csrc/autograd/generated/VariableType_0.cpp

    可以看到和 grad_fn 相关的操作trace了一个op的计算，构建了后向计算图.

## 后向计算

autograd::backward():计算output var的梯度值，调用的 run_backward()

autograd::grad() ：计算有output var和到特定input的梯度值，调用的 run_backward()

autograd::run_backward()·       g'f

    对于要求梯度的output var，获取其指向的grad_fn作为roots，是后向图的起点

    对于有input var的，获取其指向的grad_fn作为output_edges, 是后向图的终点

    调用 autograd::Engine::get_default_engine().execute(...) 执行后向计算

autograd::Engine::execute(...)

    创建 GraphTask ，记录了一些配置信息

    创建 GraphRoot ，是一个Node，把所有的roots作为其输出边，Node的apply()返回的是roots的grad【这里已经得到一个单起点的图】

    计算依赖 compute_dependencies(...)

        从GraphRoot开始，广度遍历，记录所有碰到的grad_fn的指针，并统计grad_fn被遇到的次数，这些信息记录到GraphTask中

    GraphTask 初始化：当有input var时，判断后向图中哪些节点是真正需要计算的

    GraphTask 执行

        选择CPU or GPU线程执行

        以CPU为例，调用的 autograd::Engine::thread_main(...)

autograd::Engine::thread_main(...)

    evaluate_function(...) ，输入输出的处理，调度

        call_function(...) , 调用对应的Node计算

        执行后向过程中的生成的中间grad Tensor，如果不释放，可以用于计算高阶导数；（同构的后向图，之前的grad tensor是新的输出，grad_fn变成之前grad_fn的backward，这些新的输出还可以再backward）

    具体的执行机制可以支撑单独开一个Topic分析，在这里讨论到后向图完成构建为止.
## 参考

- https://blog.csdn.net/zandaoguang/article/details/115713552

