# PyTorch的算子体系

## 主要内容
- [Tensor](#Tensor)
- [torch模块中的函数](#torch模块中的函数)
- [Tensor算子](#Tensor算子)
- [torch.nn](#torch.nn)
- [torch.nn.functional](#torch.nn.functional)
- [torch.autograd](#torch.autograd)
- [torch.multiprocessing](#torch.multiprocessing)
- [torch.cuda](#torch.cuda)
- [torch.legacy](#torch.legacy)
- [torch.utils.ffi](#torch.utils.ffi)
- [torch.utils.data](#torch.utils.data)
- [torch.utils.model_zoo](#torch.utils.model_zoo)
- [参考](#参考)

## Tensor

### Tensor的数据

### Tensor storage

### Sparse Tensor
PyTorch对于稀疏存储格式的支持,目前主要支持COO和CSR格式。
#### COO格式
采用三元组(row, col, data)(或称为ijv format)的形式来存储矩阵中非零元素的信息三个数组 row 、col 和 data 分别保存非零元素的行下标、列下标与值（一般长度相同）故 coo[row[k]][col[k]] = data[k] ，即矩阵的第 row[k] 行、第 col[k] 列的值为 data[k] 
<img src="../images/tensor_coo.gif"/>
https://zhuanlan.zhihu.com/p/188700729

优点: 非常方便转换成其他格式，如tobsr()、tocsr()、to_csc()、to_dia()、to_dok()、to_lil()等
缺点：不支持切片和算术运算操作如果稀疏矩阵仅包含非0元素的对角线，则对角存储格式(DIA)可以减少非0元素定位的信息量这种存储格式对有限元素或者有限差分离散化的矩阵尤其有效

#### CSR格式
Compressed Sparse Row Matrix:压缩稀疏行格式
csr_matrix按行对矩阵进行压缩，通过indices,indptr,data来确定矩阵。data表示矩阵中的非零数据。对于第i行而言，该行中非零元素的列索引为 indices[indptr[i]:indptr[i+1]]可以将 indptr 理解成利用其自身索引 i 来指向第 i 行元素的列索引根据[indptr[i]:indptr[i+1]]，我就得到了该行中的非零元素个数。
<img src="../images/tensor_csr.gif"/>
https://zhuanlan.zhihu.com/p/188700729
根据上图：
- 第一行：非零值的列为[0,2]，因此
    indptr[0] = 0       # 表示indices[0:2]为第一行非零值的列
    indptr[0] = 2
    indices[0] = 0
    indices[1] = 2
    data[0] = 8
    data[1] = 2
- 第二行，非零值的列为2，因此
    indptr[1] = 2       # 表示indices[2:3]为第二行非零值的列
    indptr[2] = 3   
    indices[2] = 2      # 列为2  
    data[2] = 5         # 第二行只有一个非零值
- 第三行，没有非零值，因此
    indptr[2] = 3       # 3-3=0 表示第三行非零值的个数为0
    indptr[3] = 3   
- 第四行，没有非零值，因此
    indptr[3] = 3       # 3-3=0 表示第四行非零值的个数为0
    indptr[4] = 3   
- 第五行，非零值的列为[2, 3, 4]，因此
    indptr[4] = 3       # 6-3=3 表示第四行非零值的个数为3
    indptr[5] = 6       
    indices[3] = 2
    indices[4] = 3
    indices[5] = 4
    data[3] = 7
    data[4] = 1
    data[5] = 2

    

### DLPack
DLPack是一个内存张量结构的开放标准，支持不同框架之间的张量转换。
PyTorch使用torch.utils.dlpack实现DLPack与tensor之间的转换。

## torch模块中的函数

### Tensor操作
```Python
torch.is_tensor(obj)        # obj是否为PyTorch tensor对象
torch.is_storage(obj)       # obj是否为PyTorch storage对象
torch.is_complex(input)     # input是否为complex类型（torch.complex64或torch.complex128）
torch.is_conj(input)        # input是否共轭矩阵，只检查相关标志位
torch.is_floating_point(input)  # input是否为float类型，包括torch.float64, torch.float32, torch.float16以及torch.bfloat16.
torch.is_nonzero(input)     # 判断input是否只有1个元素并且转换后值不为0.
torch.set_default_dtype(d)  # 设置缺省浮点类型，包括torch.float32和torch.float64 as inputs.
torch.get_default_dtype()   # 获取缺省浮点类型
torch.set_default_tensor_type(t)    # 设置缺省tensor浮点类型.
torch.numel(input)          # 返回input tensor中元素个数
torch.set_printoptions(...) # 设置打印格式
torch.set_flush_denormal(mode)  # 设置非规格化浮点数模式
```

### 创建Tensor
```Python
torch.tensor(...)           # 通过拷贝构造创建tensor
torch.sparse_coo_tensor(...)    # 构建COO格式的系数tensor
torch.asarray(...)          # 将对象转换成tensor
torch.as_tensor(...)        # 将数据转换成tensor
tarch.as_strided(...)       # 基于已有tensor创建新的view，并指定stride等参数
torch.from_numpy(ndarray)   # 基于numpy.ndarray创建tensor
torch.from_dlpack(ext_tensor)   # 基于dlpack张量创建tensor
torch.frombuffer(...)       # 基于python buffer创建一堆的tensor
torch.zeros(...)            # 创建元素全为8的tensor
torch.zeros_like(...)       # 使用已有tensor的shape，创建元素全为8的tensor
torch.ones(...)             # 创建元素全为1的tensor
torch.ones_like(...)        # 使用已有tensor的shape，创建元素全为1的tensor
torch.arange(...)           # 创建等差数组形式的tensor
torch.range(...)            # 创建等差数组形式的tensor
torch.linspace<...>         # 创建等差数组形式的tensor
torch.logspace(...)         # 创建指数数组形式的tensor，其指数为等差数列
torch.eye(...)              # 创建对角矩阵
torch.empty(...)            # 创建未初始化的tensor
torch.empty_strided(...)    # 使用已有tensor的shape，创建未初始化的tensor
torch.full(...)             # 基于指定的值，创建tensor
torch.full_like(...)        # 使用已有tensor的shape，创建元素为指定值的tensor
torch.quantize_per_tensor(...)  # 将tensor转换成量化的格式
torch.quantize_per_channel(...) # 将tensor转换成量化的格式
torch.dequantize(tensor)    # 将量化的tensor还原为普通的fp32 tensor
torch.complex(...)          # 创建复数tensor
torch.polar(...)            # 根据极坐标参数创建笛卡尔坐标的tensor
torch.heaviside(...)        # 根据heaviside step函数，基于给定值创建tensor
```

### 索引、切片及连接等
```Python
torch.adjoint(Tensor)       # 创建tensor共轭视图
torch.argwhere(input)       # 根据tensor中非零值的坐标创建新的tensor
torch.cat(...)              # 在指定维度上拼接多个tensor
torch.concat(...)           # 同上
torch.concatenate(...)      # 同上
torch.conj(input)           # 创建tensor共轭视图
torch.chunk(...)            # 分割指定的tensor
torch.dsplit(...)           # 在多个维度上分割指定的tensor
torch.column_stack(...)     # 堆叠多个tensor
torch.dstack(...)           # 堆叠多个tensor
torch.gather(...)           # 根据位置提取tensor内的值
torch.hsplit(...)           # 在多个维度上分割指定的tensor
torch.hstack(...)           # 堆叠多个tensor
torch.index_add(...)        # 根据指定index和权重进行求和
torch.index_reduce(...)     # 根据指定index和权重进行累加
torch.index_select(...)     # 根据指定index对tensor进行过滤
torch.masked_select(...)    # 根据指定index从tensor中提取元素
torch.movedim(...)          # 转换维度的次序
torch.moveaxis(...)         # 同上
torch.narrow(...)           # 根据指定维度截取tensor
torch.nonzero(...)          # 提取非零值的index
torch.permute(...)          # 重排tensor的维度并创建新的view
torch.reshape(...)          # 根据给定shape和已有tensor的数据生成新的tensor
torch.row_stack(...)        # 按行堆叠tensor
torch.select(...)           # 根据指定维度和索引提取tensor
torch.scatter(...)          # 将指定的值嵌入到tensor中
torch.diagonal_scatter(...) # 将指定的值嵌入到tensor对角元素中
torch.select_scatter(...)   # 将指定的值嵌入到tensor指定位置
torch.slice_scatter(...)    # 将指定的值嵌入到tensor指定维度
torch.scatter_add(...)      # 按指定维度和位置相加
torch.scatter_reduce(...)   # 使用指定规约方法进行计算
torch.split(...)            # 按指定维度分割tensor
torch.squeeze(...)          # 去除大小为1的维度
torch.stack(...)            # 堆叠同样大小的tensor并形成新的维度
torch.swapaxes(...)         # 同numpy swapaxes, 同transpose
torch.swapdims(...)         # 同上
torch.t(input)              # 2维转置
torch.take(...)             # 从头tensor中提取元素
torch.take_along_dim(...)   # 从tensor中提取元素值
torch.tensor_split(...)     # 对tensor进行分割
torch.tile(...)             # 反复堆叠同一个tensor形成新的tensor
torch.transpose(...)        # 在指定维度上做转置
torch.unbind(...)           # 删除指定的维度
torch.unsqueeze(...)        # 在指定位置增加维度
torch.vsplit(...)           # 分割tensor
torch.vstack(...)           # 按行堆叠多个tensor
torch.where(...)            # 根据条件从tensor中筛选元素
```

### 随机数相关算子
```Python
torch.Generator(...)        # 创建随机数发生器
torch.seed()                # 生成随机数种子
torch.manual_seed(seed)     # 设置随机数种子
torch.initial_seed(...)     # 获取初始随机数种子
torch.get_rng_state(...)    # 获取随机数发生器状态
torch.set_rng_state(...)    # 设置随机数发生器状态
torch.bernoulli(...)        # 根据指定概率生成伯努利分布
torch.multinomial(...)      # 根据指定权重多次多项式分布取样
torch.normal(...)           # 根据指定均值方差生成正态分布
torch.poisson(...)          # 生成泊松分布
torch.rand(...)             # 生成[0,1)区间的均匀分布
torch.rand_like(...)        # 生成[0,1)区间的均匀分布
torch.randint(...)          # 生成指定区间内均匀分布的整数序列
torch.randint_like(...)     # 生成指定区间内均匀分布的整数序列
torch.randn(...)            # 生成均值为0方差为1的正态分布
torch.randn_like(...)       # 生成均值为0方差为1的正态分布
torch.randperm(...)         # 生成0到n-1的随机排列
torch.quasirandom.SobolEngine(...) # 创建低差异分布Sobol序列的发生器
```

### 序列号
```Python
torch.save(...)             # 将对象序列化为一个文件
torch.load(...)             # 从文件反序列化为一个对象
```
### 并行操作
```Python
torch.get_num_threads()     # 获取并行CPU操作的线程数
torch.set_num_threads(int)  # 设置并行CPU操作的线程数
torch.get_num_interop_threads() # 获取CPU上算子间并行的线程数
torch.set_num_interop_threads(int)  # 设置CPU上算子间并行的线程数
```

### 

## Tensor算子
## torch.nn的算子
## torch.nn.functional
## torch.optim
## torch.autograd
## torch.multiprocessing
## torch.cuda
## torch.legacy
## torch.utils.ffi
## torch.utils.data
## torch.utils.model_zoo

## 使用算子的注意事项

### 复现性
https://pytorch.org/docs/stable/notes/randomness.html
