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
torch.sparse_coo_tensor(...)
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