
# 深度学习性能优化

## 主要内容
- [性能优化介绍][#性能优化介绍]
- [硬件加速技术-CPU][#硬件加速技术-CPU]
    - [Intel® Advanced Matrix Extensions][#Intel® Advanced Matrix Extensions]
- [模型保存及加载][#模型保存及加载]
    - [PyTorch模型存储的格式][#PyTorch模型存储的格式]
    - [PyTorch模型与ONNX模型的转换][#PyTorch模型与ONNX模型的转换]
- 使用TensorRT
- 算子融合
- 量化
- 剪枝
- 混合精度训练

## 性能优化介绍

深度神经网络的计算有以下几个特点：
- 计算量大，尤其是在当今大模型成为流行趋势的年代
- 并行度高，网络中的计算包含大量的向量和矩阵的计算，如
- 缺省数据类型是32位浮点类型，并且对精度有一定的容忍性，因此在一定的情况下，可以使用bfloat16或者int8进行计算
- 网络结构有一定的裁剪容忍度，剪掉部分连接对整体预测精度的影响不大

## 硬件加速技术-CPU

在Linux操作系统中，可以使用命令"cat /proc/cpuinfo"来查看CPU的型号及其所支持的指令集

```Bash
# cat /proc/cpuinfo

flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat avx512vbmi umip pku ospke avx512_vpopcntdq la57 rdpid arch_capabilities
```

输出中flag这一项指明了CPU所支持的硬件特性：

- SSE
- AVX
- AMX

<table>
<tr>
<td>h1</td><td>h2</td>
</tr>
</table>

|指令集| 	条| 	Date| 	ICPU| 	IDate| 	ACPU| 	ADate| 	Memo|
|-------|----|----------|-------|--------|-------|-------|-------|
|MMX| 	57| 	1996-10-12| 	Pentium MMX(P55C)| 	1996-10-12| 	K6| 	1997-4-1| 	MultiMedia eXtension|
SSE| 	70| 	1999-5-1| 	Pentium III(Katmai)| 	1999-5-1| 	Athlon XP| 	2001-10-9| 	Streaming SIMD| Extensions|
SSE2| 	144| 	2000-11-1| 	Pentium 4(Willamette)| 	2000-11-1| 	Opteron| 	2003-4-22|| 	 
SSE3| 	13| 	2004-2-1| 	Pentium 4(Prescott)| 	2004-2-1| 	Athlon 64| 	2005-4-1|| 	 
SSSE3| 	16| 	2006-1-1| 	Core| 	2006-1-1| 	Fusion(Bobcat)| 	2011-1-5| 	最早出现在Tejas核心（功耗过高而取消）
SSE4.1| 	47| 	2006-9-27| 	Penryn| 	2007-11-1| 	Bulldozer| 	2011-9-7| 	 
SSE4.2| 	7| 	2008-11-17| 	Nehalem| 	2008-11-17| 	Bulldozer| 	2011-9-7| 	 
SSE4a| 	4| 	2007-11-11| |	  |	  	K10| 	2007-11-11| 	K10还加了 POPCNT 与 LZCNT 指令
SSE5| 	 | 	2007-8-30| 	  	|||  	  	  	|被AVX搅局。后来XOP/FAM4/CVT16
AVX| 	 | 	2008-3-1| 	Sandy Bridge| 	2011-1-9| 	Bulldozer| 	2011-9-7| 	Advanced Vector Extensions
AVX2| 	 | 	2011-6-13| 	Haswell| 	2013-4-1| 	  	  	 
AES| 	7| 	2008-3-1| 	Westmere| 	2010-1-7| 	Bulldozer| 	2011-9-7| 	Advanced Encryption Standard
3DNowPrefetch| 	2| 	2010-8-1| 	  ||	  	K6-2| 	1998-5-28| 	2010年8月放弃3DNow!，仅保留2条预取
3DNow!| 	21| 	1998-1-1| 	  | |	  	K6-2| 	1998-5-28| 	 
3DNow!+| 	|  	1999-6-23| 	  ||	  	Athlon| 	1999-6-23| 	Enhanced 3DNow!. 共52条？
MmxExt| 	|  	 | 	 || 	  	Athlon| 	1999-6-23| 	Extensions MMX
3DNow! Pro| 	||||  	  	  	  	Athlon XP| 	2001-10-9| 	3DNow! Professional.兼容SSE
POPCNT| 	1| 	2007-11-11| 	||  	  	K10| 	2007-11-11| 	 
ABM| 	1| 	2007-11-11| 	||  	  	K10| 	2007-11-11| 	advanced bit manipulation. LZCNT
CLMUL| 	5| 	2008-5-1| 	Westmere| 	2010-1-7| 	Bulldozer| 	2011-9-7| 	PCLMULQDQ等
F16C| 	|  	2009-5-1| 	Ivy Bridge| 	2012-4-1| 	Bulldozer| 	2011-9-7| 	CVT16|
FAM4| 	 | 	2009-5-1| 	 || 	  	Bulldozer| 	2011-9-7| 	 
XOP| 	 | 	2009-5-1| 	  ||	  	Bulldozer| 	2011-9-7| 	 

指令集：指令集名。
条：指令条数。
Date：公布日期。
ICPU：Intel最早支持该指令集的CPU。
IDate：ICPU的发售日期。
ACPU：AMD最早支持该指令集的CPU。
ADate：ACPU的发售日期。
Memo：备注。

参考https://www.cnblogs.com/zyl910/archive/2012/02/26/x86_simd_table.html


基于CPU的加速库：

- MKLDNN
- ONEDNN

### Intel® Advanced Matrix Extensions

## 模型保存及加载

### PyTorch模型存储的格式

### PyTorch模型与ONNX模型的转换

## BFloat16

BFloat16 (Brain Floating Point)是一种16bit的浮点数格式，动态表达范围和float32是一样的，但是精度低。下一代的Xeon Sapphire Rapids上面可以使用AMX(Advanced Matrix Extensions)对卷积和矩阵乘的操作在BFloat16上进行加速，吞吐量比Float32高一个数量级。

这里主要介绍在PyTorch上面优化BFloat16原生算子的一些小技巧，侧重性能优化方面，不介绍BFloat16训练中涉及的调参问题。

## 参考
- PyTorch CPU性能优化（四）：BFloat16 https://zhuanlan.zhihu.com/p/499979372