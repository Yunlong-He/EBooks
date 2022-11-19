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
torch.scatter(...)          # 
```

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
