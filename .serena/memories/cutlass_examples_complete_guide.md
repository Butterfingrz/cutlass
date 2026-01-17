# CUTLASS Examples 完整指南

## 概述

CUTLASS包含84个编号的examples，形成了从基础到前沿的完整GPU计算技术栈学习和开发体系。这些examples按照技术复杂度和架构代际进行分层组织，为不同层次的开发者提供了清晰的学习路径。

⚠️ **重要提醒**: 这些examples专门用于**演示CUTLASS功能**，可能**未针对性能基准测试进行优化**。如需准确的性能测量，请使用**CUTLASS Profiler**。

## 技术分层架构

### 基础层 (00-19): 核心功能与工具

#### 基础GEMM系列 (00-09)
- **00_basic_gemm**: 基础单精度GEMM操作
- **01_cutlass_utilities**: CUTLASS工具库的使用(张量分配和初始化)
- **02_dump_reg_smem**: 调试工具(打印寄存器和共享内存内容)
- **03_visualize_layout**: 布局函数可视化工具
- **04_tile_iterator**: 内存tile迭代器演示
- **05_batched_gemm**: 批处理strided GEMM操作
- **06_splitK_gemm**: Split-K并行reduce内核
- **07_volta_tensorop_gemm**: Volta Tensor Core混合精度GEMM
- **08_turing_tensorop_gemm**: Turing Tensor Core整型GEMM
- **09_turing_tensorop_conv2dfprop**: Turing Tensor Core整型隐式GEMM卷积

#### 复杂数据类型系列 (10-19)
- **10_planar_complex**: 平面复数GEMM内核
- **11_planar_complex_array**: 批处理特定问题规模的平面复数内核
- **12_gemm_bias_relu**: GEMM与bias和ReLU融合
- **13_two_tensor_op_fusion**: 两个GEMM或卷积在一个内核中融合
- **14_ampere_tf32_tensorop_gemm**: FP32 GEMM with隐式TF32转换
- **15_ampere_sparse_tensorop_gemm**: 稀疏Tensor Core使用演示
- **16_ampere_tensorop_conv2dfprop**: NHWC布局前向卷积
- **17_fprop_per_channel_bias**: 卷积与per channel bias和ReLU融合
- **18_ampere_fp64_tensorop_affine2_gemm**: Affine-2 GEMM演示
- **19_tensorop_canonical**: 使用tensor core的标准GEMM

### 算法进阶层 (20-39): 高级算法与优化

#### 特殊数据类型与算法 (20-29)
- **20_simt_canonical**: 使用SIMT的标准GEMM
- **21_quaternion_gemm**: 四元数GEMM计算
- **22_quaternion_conv**: 四元数卷积
- **23_ampere_gemm_operand_reduction_fusion**: GEMM操作数K维reduction融合
- **24_gemm_grouped**: 不同问题规模的批处理GEMM操作
- **25_ampere_fprop_mainloop_fusion**: 激活函数的per channel scale+bias+relu融合到fprop mainloop
- **26_ampere_wgrad_mainloop_fusion**: 激活函数的per channel scale+bias+relu融合到wgrad mainloop
- **27_ampere_3xtf32_fast_accurate_tensorop_gemm**: 使用TF32操作模拟快速准确SGEMM
- **28_ampere_3xtf32_fast_accurate_tensorop_fprop**: 使用TF32操作模拟快速准确FP32卷积
- **29_ampere_3xtf32_fast_accurate_tensorop_complex_gemm**: 使用TF32操作模拟快速准确CGEMM

#### 专业矩阵操作 (30-39)
- **30_wgrad_split_k**: conv2d梯度weight计算与split-K
- **31_basic_syrk**: 对称rank-K更新
- **32_basic_trmm**: 三角矩阵-矩阵乘法
- **33_ampere_3xtf32_tensorop_symm**: FP32模拟的对称矩阵-矩阵乘法
- **34_transposed_conv2d**: 2D转置卷积(反卷积)，使用CUTLASS conv2d Dgrad内核
- **35_gemm_softmax**: GEMM与Softmax在混合精度下的融合
- **36_gather_scatter_fusion**: GEMM前的gather和GEMM后的scatter融合
- **37_gemm_layernorm_gemm_fusion**: gemm->layernorm->gemm融合为一个内核
- **38_syr2k_grouped**: 不同问题规模的批处理SYR2K操作
- **39_gemm_permute**: 输出结果置换为重塑张量的批处理GEMM操作

### 高级特性层 (40-59): 现代架构与Python接口

#### 注意力机制与Python接口 (40-49)
- **40_cutlass_py**: CUTLASS Python接口演示
- **41_multi_head_attention**: 非固定序列长度输入的注意力示例
- **42_ampere_tensorop_group_conv**: 使用tensor core的群卷积内核
- **43_ell_block_sparse_gemm**: Block-Ell稀疏GEMM
- **44_fused_multi_head_attention**: 使用共享内存的融合多头注意力(固定&可变)
- **45_dual_gemm**: 共享相同左输入矩阵的两个GEMM融合
- **46_depthwise_simt_conv2dfprop**: 使用SIMT指令的深度卷积2D
- **47_ampere_gemm_universal_streamk**: Stream-K并行分解对比"classic data-parallel"和"Split-K"分解

#### Hopper架构特性 (48-59)
- **48_hopper_warp_specialized_gemm**: 使用CUTLASS 3.0 API的Hopper架构简单tensorop GEMM
- **49_hopper_gemm_schedules_with_collective_builder**: Hopper GEMM集合操作构建器演示，展示构建器API和CUTLASS 3.0支持的各种内核调度
- **50_hopper_gemm_with_epilogue_swizzle**: Hopper GEMM with自定义集合mainloop和自定义向量化epilogue
- **51_hopper_gett**: Hopper GETT演示，展示由于CUTLASS 3.0的统一微内核和CuTe的分层布局，GETT运行的便利性
- **52_hopper_gather_scatter_fusion**: Hopper example that融合GEMM前的gather和GEMM后的scatter到同一内核
- **53_hopper_gemm_permute**: Hopper演示张量置换操作与GEMM内核的融合
- **54_hopper_fp8_warp_specialized_gemm**: Hopper FP8 GEMM内核实例化和运行
- **55_hopper_mixed_dtype_gemm**: Hopper GEMM with不同A和B数据类型，使用CUTLASS 3.x API的DL内核与融合反量化
- **56_hopper_ptr_array_batched_gemm**: Hopper指针数组批处理GEMM，使用CUTLASS 3.x API
- **57_hopper_grouped_gemm**: Hopper分组GEMM，使用CUTLASS 3.x API

#### 特殊架构支持 (58-59)
- **58_ada_fp8_gemm**: Ada GEMM内核，通过CUTLASS 2.x API针对Ada FP8 tensor core
- **59_ampere_gather_scatter_conv**: CuTe和CUTLASS 3.x基于Ampere卷积fprop内核，能够操作仿射和gather/scatter张量

### 前沿层 (70-84): Blackwell架构专门功能

#### Blackwell基础功能 (70-79)
- **70_blackwell_gemm**: 针对NVIDIA Blackwell SM100 Tensor Core MMA的简单密集GEMM，使用CUTLASS 3.x API
- **71_blackwell_gemm_with_collective_builder**: Blackwell SM100 GEMM演示兼容的mainloop+epilogue构建器调度和epilogue访问者树(EVT)构造
- **72_blackwell_narrow_precision_gemm**: 针对NVIDIA Blackwell SM100 Tensor Core MMA的块缩放密集GEMM，使用CUTLASS 3.x API
- **73_blackwell_gemm_preferred_cluster**: Blackwell SM100 GEMM内核与首选cluster特性
- **74_blackwell_gemm_streamk**: Blackwell SM100 GEMM内核使用Stream-K调度器
- **75_blackwell_grouped_gemm**: Blackwell SM100分组GEMM内核
- **76_blackwell_conv**: 针对NVIDIA Blackwell SM100 Tensor Core MMA的简单卷积(fprop/dgrad/wgrad)，使用CUTLASS 3.x API
- **77_blackwell_fmha**: Blackwell SM100 FMHA内核
- **78_blackwell_emulated_bf16x9_gemm**: Blackwell SM100 FastFP32(使用BF16模拟SGEMM)内核
- **79_blackwell_geforce_gemm**: Blackwell SM120 MMA内核，针对GeForce RTX 50系列CUDA Core

#### Blackwell高级功能 (80-84)
- **80_blackwell_geforce_sparse_gemm**: Blackwell SM120稀疏MMA内核，针对GeForce RTX 50系列CUDA Core
- **83_blackwell_sparse_gemm**: Blackwell SM100稀疏GEMM内核
- **84_blackwell_narrow_precision_sparse_gemm**: Blackwell块缩放SM100稀疏GEMM内核

## 架构演进路径

### GPU架构支持映射
- **Volta (SM 7.0)**: Examples 07
- **Turing (SM 7.5)**: Examples 08-09
- **Ampere (SM 8.0+)**: Examples 14-33, 42, 59
- **Hopper (SM 9.0)**: Examples 48-57
- **Ada Lovelace (SM 8.9)**: Example 58
- **Blackwell (SM 10.0)**: Examples 70-84

### 关键技术演进
1. **内存层次掌握**: 00-09(共享内存) → 40s(TMA) → 70s(分布式共享内存)
2. **融合演进**: 12(简单bias) → 30s(复杂epilogue) → 50s(flash attention)
3. **稀疏性发展**: 15(结构化) → 43(2:4) → 83(可变模式)

## 学习路径建议

### 初学者路径
1. **基础理解**: 00-03 (GEMM基础)
2. **工具掌握**: 01-02 (调试和实用工具)
3. **架构入门**: 根据目标硬件选择07(Volta), 08-09(Turing), 14(Ampere)

### 中级开发者路径
1. **融合操作**: 12, 35, 36 (学习融合模式)
2. **复杂数据**: 10-11 (复数), 21-22 (四元数)
3. **Python集成**: 40 (Python接口快速原型)

### 高级开发者路径
1. **注意力机制**: 41(基础) → 44(融合) → 77(FMHA)
2. **稀疏计算**: 15 → 43 → 83-84
3. **最新架构**: 48-57(Hopper) → 70-84(Blackwell)

## 开发最佳实践

### 架构选择策略
- **Ampere/Hopper**: Examples 40-59包含生产就绪模式
- **性能优化**: 27(FP8) → 55(FP8快速累积)用于推理优化
- **内存效率**: 50s示例展示TMA和分布式共享内存使用

### 构建和测试建议
1. **依赖矩阵**: Examples 70+需要CUDA 12.4+和特定驱动版本
2. **增量测试**: 从匹配部署架构的examples开始
3. **性能基线**: 运行examples 00, 07(性能分析)和架构特定基准

## 缺失编号分析

### 60-69范围
- 可能为Ada Lovelace(SM89)特定功能保留
- 或者Hopper扩展功能的未来占位符

### 85-92范围  
- 可能为未来Blackwell功能保留
- 或下一代架构(Rubin?)的占位符
- NVIDIA通常为快速迭代保留编号空间

## 专家洞察

### 隐藏复杂性
1. **版本依赖**: Examples 70+需要特定CUDA版本和驱动
2. **构建系统**: CMake配置在不同examples范围间差异显著
3. **性能可移植性**: 针对Hopper优化的代码可能在旧架构上表现不佳

### 生产使用建议
- **注意力机制**: 41(基础) → 50(flash) → 53(分组查询)
- **混合精度**: 27(FP8) → 55(FP8快速累积)用于推理优化
- **稀疏计算**: 根据稀疏模式选择15, 43, 或83-84

## 相关资源

### 补充材料
- **CuTe Examples**: `cutlass/examples/cute/` - 不依赖CUTLASS的CuTe特性展示
- **Python Examples**: `cutlass/examples/python/` - Python接口示例
- **Unit Tests**: `cutlass/test/unit/cute/core/` - CuTe核心测试案例

### 文档链接
- [CUTLASS Profiler](../tools/profiler/) - 性能测量工具
- [Python Interface](../python/README.md) - Python绑定文档