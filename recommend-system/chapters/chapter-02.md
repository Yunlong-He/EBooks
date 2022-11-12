
# 信息熵相关算法

## ID3 算法

信息增益的计算方式仍然如下：

$$g(D, A) = H(D) - H(D|A)$$

$$H(D) = -\sum_{k=1}^K\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|}$$
