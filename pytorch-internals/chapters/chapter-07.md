
# 自动微分

自动微分一直被视为深度学习框架的核心能力，在训练深度学习神经网络的时候，网络的参数需要根据输出端的梯度不断的调整，如果没有自动微分，我们必须为每个算子指定一个根据梯度调节参数的计算方法，由于这种计算方法不像前向转播中算子的公式那样直接明了，手工实现就成为了几乎不可能完成的任务。最早的深度学习框架Theano吸引人的一个主要优点就是支持自动微分，因此在现今流行的深度学习框架中，自动微分已经成为了必不可少的内置功能。


## 自动微分的理论基础

在了解自动微分之前，我们先从优化的角度看一下参数和梯度的关系，这也是深度学习的目标。

考虑下面这个公式，这是典型的线性回归的公式，我们需要根据输出与实际值的差异调整系数$w$及截距$b$：
$$y=w*x + b$$

根据微分原理我们知道：
$$\frac{\partial{y}}{\partial{w}} = x$$
$$\frac{\partial{y}}{\partial{b}} = 1$$

根据上面的式子，在微小的取值范围内，为了调整$w$，可以这样计算：
$$\mathrm{d}w = x * \mathrm{d}y$$
其中$\mathrm{d}y$ 就是输出与实际值的差异。在实际计算中，由于$\mathrm{d}y$的值不会很小，我们会加一个比较小的系数$\alpha$来缓慢调整$w$:
$$\mathrm{d}w = \alpha * x * \mathrm{d}y$$

同理，对于另一个算子：

$$y=w*x^2$$

我们可以计算得到：
$$\mathrm{d}w = \alpha * x^2 * \mathrm{d}y$$

下面我们看看自动微分是怎样在PyTorch中实现的，在探究之前，我们先关注几个问题：
- PyTorch中的计算图是怎样构建的？
- 反向传播的流程是什么样的？

### 计算图及反向传播

在计算图中，autograd会记录所有的操作，并生成一个DAG（有向无环图），其中输出的tensor是根节点，输入的tensor是叶子节点，根据链式法则，梯度的计算是从根结点到叶子节点的逐步计算过程。

在前向阶段，autograd同时做两件事：
- 根据算子计算结果Tensor
- 维护算子的梯度函数

在反向阶段，当.backward()被调用时，autograd:
- 对于节点的每一个梯度函数，计算相应节点的梯度
- 在节点上对梯度进行累加，并保存到节点的.grad属性上
- 根据链式法则，按照同样的方式计算，一直到叶子节点

对于一个简单的例子：
```python
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3 - b**2

```
下图是对应的计算图，其中的函数代表梯度计算函数：
<img src='../images/dag_autograd.png'/>



## 自动微分相关的核心数据结构

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
// c10/core/TensorImpl.h

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
 autograd_meta_表示 Variable 中关于计算梯度的元数据信息，AutogradMetaInterface 是一个接口，有不同的子类，这里的 Variable 对象的梯度计算的元数据类型为 AutogradMeta，其部分成员为 
```C++
// torch/csrc/autograd/variable.h

struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;
  std::shared_ptr<ForwardGrad> fw_grad_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  bool requires_grad_;
  bool retains_grad_;
  bool is_view_;
  uint32_t output_nr_;

  // ...
}

```

grad_ 表示反向传播时，关于当前 Variable 的梯度值。grad_fn_ 是用于计算非叶子Variable的梯度的函数，比如 AddBackward0对象用于计算result这个Variable 的梯度。对于叶子Variable，此字段为 None。grad_accumulator_ 用于累加叶子 Variable 的梯度累加器，比如 AccumulateGrad 对象用于累加 self的梯度。对于非叶 Variable，此字段为 None。output_nr_ 表示当前 Variable 是 计算操作的第一个输出，此值从 0 开始。

可以看到，grad_fn_和grad_accumulator_都是Node的指针，这是因为在计算图中，算子的C++类型是Node，不同的算子的实现都是Node的子类。

Node是由上一级的Node创建的

```C++
// torch/include/torch/csrc/autograd/function.h

struct TORCH_API Node : std::enable_shared_from_this<Node> {
 public:
  /// Construct a new `Node` with the given `next_edges`
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit Node(
      uint64_t sequence_nr,
      edge_list&& next_edges = edge_list())
      : sequence_nr_(sequence_nr),
      next_edges_(std::move(next_edges)) {

    for (const Edge& edge: next_edges_) {
      update_topological_nr(edge);
    }

    if (AnomalyMode::is_enabled()) {
      metadata()->store_stack();

      assign_parent();
    }

    // Store the thread_id of the forward operator.
    // See NOTE [ Sequence Numbers ]
    thread_id_ = at::RecordFunction::currentThreadId();
  }



  /// Evaluates the function on the given inputs and returns the result of the
  /// function call.
  variable_list operator()(variable_list&& inputs) {
    // ...
    return apply(std::move(inputs));
  }

  uint32_t add_input_metadata(const at::Tensor& t) noexcept {
    // ...
  }

  void add_next_edge(Edge edge) {
    update_topological_nr(edge);
    next_edges_.push_back(std::move(edge));
  }

 protected:
  /// Performs the `Node`'s actual operation.
  virtual variable_list apply(variable_list&& inputs) = 0;

  variable_list traced_apply(variable_list inputs);


  const uint64_t sequence_nr_;

  uint64_t topological_nr_ = 0;


  mutable bool has_parent_ = false;


  uint64_t thread_id_ = 0;


  std::mutex mutex_;

  edge_list next_edges_;

  PyObject* pyobj_ = nullptr; 

  std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;

  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;

  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;

  at::SmallVector<InputMetadata, 2> input_metadata_;
};

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
- https://zhuanlan.zhihu.com/p/111239415
- https://zhuanlan.zhihu.com/p/138203371
