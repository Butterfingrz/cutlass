# CUTLASS Examples 开发场景应用指南

## 典型开发场景与推荐Examples

### 场景1: 新手入门CUTLASS开发

#### 学习路径
1. **起步阶段**: 00_basic_gemm
   - 理解CUTLASS基本API结构
   - 学习参数配置和内核启动
   - 掌握性能测量方法

2. **工具掌握**: 01_cutlass_utilities + 02_dump_reg_smem
   - 学习张量分配和初始化
   - 掌握调试工具使用
   - 理解内存布局和数据流

3. **架构理解**: 03_visualize_layout + 04_tile_iterator
   - 深入理解CUTLASS布局系统
   - 掌握tile和迭代器概念
   - 为高级开发打基础

#### 实践建议
```bash
# 推荐学习顺序
cd cutlass/examples
make 00_basic_gemm && ./00_basic_gemm/00_basic_gemm
make 01_cutlass_utilities && ./01_cutlass_utilities/01_cutlass_utilities  
make 02_dump_reg_smem && ./02_dump_reg_smem/02_dump_reg_smem
```

### 场景2: 深度学习模型加速

#### 训练加速Examples
- **基础训练**: 14_ampere_tf32_tensorop_gemm (FP32训练)
- **混合精度**: 07_volta_tensorop_gemm (FP16+FP32)
- **前向传播**: 16_ampere_tensorop_conv2dfprop (卷积优化)
- **反向传播**: 26_ampere_wgrad_mainloop_fusion (梯度计算)
- **注意力机制**: 41_multi_head_attention → 44_fused_multi_head_attention

#### 推理优化Examples
- **量化推理**: 54_hopper_fp8_warp_specialized_gemm (FP8推理)
- **批处理推理**: 56_hopper_ptr_array_batched_gemm (动态批处理)
- **融合优化**: 35_gemm_softmax (GEMM+Softmax融合)
- **稀疏推理**: 83_blackwell_sparse_gemm (稀疏矩阵加速)

#### 开发流程
1. **原型开发**: 使用40_cutlass_py快速验证算法
2. **性能优化**: 参考47_ampere_gemm_universal_streamk并行策略
3. **生产部署**: 采用48-57 Hopper系列的生产级优化

### 场景3: 自定义GEMM内核开发

#### 开发阶段Examples
1. **理解基础**: 19_tensorop_canonical + 20_simt_canonical
   - 掌握标准GEMM实现模式
   - 理解Tensor Core vs SIMT差异

2. **学习融合**: 12_gemm_bias_relu → 35_gemm_softmax → 37_gemm_layernorm_gemm_fusion
   - 渐进式学习融合技术
   - 理解epilogue定制方法

3. **高级定制**: 49_hopper_gemm_schedules_with_collective_builder + 50_hopper_gemm_with_epilogue_swizzle
   - 使用集合构建器API
   - 自定义mainloop和epilogue

#### 自定义模式参考
```cpp
// 基于Example 49的模式
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

// 基于Example 50的模式  
using EpilogueOp = cutlass::epilogue::fusion::Sm90EVT<...>;
```

### 场景4: 高性能计算应用

#### 科学计算Examples
- **线性方程求解**: 31_basic_syrk + 32_basic_trmm (对称和三角矩阵)
- **复数计算**: 10_planar_complex + 11_planar_complex_array (科学计算常见)
- **四元数计算**: 21_quaternion_gemm + 22_quaternion_conv (图形学/物理)
- **高精度计算**: 27_ampere_3xtf32_fast_accurate_tensorop_gemm (数值稳定性)

#### 并行优化Examples
- **负载均衡**: 47_ampere_gemm_universal_streamk (Stream-K算法)
- **分组计算**: 24_gemm_grouped + 57_hopper_grouped_gemm
- **分布式计算**: 74_blackwell_gemm_streamk (最新调度策略)

### 场景5: 移动端和边缘设备优化

#### 量化和压缩Examples
- **整型量化**: 08_turing_tensorop_gemm (INT8/INT4)
- **混合精度**: 55_hopper_mixed_dtype_gemm (A/B不同精度)
- **窄精度**: 72_blackwell_narrow_precision_gemm (块缩放量化)
- **稀疏化**: 15_ampere_sparse_tensorop_gemm (结构化稀疏)

#### 内存优化Examples
- **内存效率**: 52_hopper_gather_scatter_fusion (减少内存访问)
- **数据重用**: 36_gather_scatter_fusion (优化数据流)

### 场景6: 研究和算法探索

#### 前沿架构Features
- **Hopper新特性**: 48-61全系列 (TMA, WGMMA, FP8)
- **Blackwell探索**: 70-84全系列 (最新架构特性)
- **Ada特性**: 58_ada_fp8_gemm (Ada Lovelace FP8)

#### 实验性算法
- **注意力变体**: 61_hopper_gemm_with_topk_and_softmax (Top-K注意力)
- **稀疏模式**: 43_ell_block_sparse_gemm → 83_blackwell_sparse_gemm
- **融合策略**: 36_gather_scatter_fusion → 52_hopper_gather_scatter_fusion

### 场景7: 生产环境部署

#### 稳定性优先Examples
- **成熟架构**: 14-39 Ampere系列 (生产验证)
- **错误处理**: 02_dump_reg_smem (调试支持)
- **性能监控**: 使用CUTLASS Profiler配合examples

#### 可扩展性Examples
- **批处理**: 05_batched_gemm → 56_hopper_ptr_array_batched_gemm
- **分组操作**: 24_gemm_grouped → 57_hopper_grouped_gemm → 75_blackwell_grouped_gemm

## 架构迁移指南

### Volta → Ampere迁移
- **起始**: 07_volta_tensorop_gemm
- **目标**: 14_ampere_tf32_tensorop_gemm
- **关键差异**: TF32支持，更大Tensor Core
- **迁移策略**: 利用15_ampere_sparse_tensorop_gemm的稀疏性

### Ampere → Hopper迁移  
- **起始**: 14-47 Ampere系列
- **目标**: 48-61 Hopper系列
- **关键差异**: WGMMA, TMA, FP8原生支持
- **迁移重点**: 49_hopper_gemm_schedules_with_collective_builder

### Hopper → Blackwell迁移
- **起始**: 48-61 Hopper系列  
- **目标**: 70-84 Blackwell系列
- **关键差异**: 增强稀疏性，窄精度优化
- **迁移重点**: 71_blackwell_gemm_with_collective_builder

## 性能调优策略

### 基础调优(所有架构)
1. **起始分析**: 00_basic_gemm建立baseline
2. **并行策略**: 06_splitK_gemm vs 47_ampere_gemm_universal_streamk
3. **内存优化**: 03_visualize_layout + 04_tile_iterator

### 架构特定调优

#### Ampere调优路径
1. **TF32优化**: 27_ampere_3xtf32_fast_accurate_tensorop_gemm
2. **融合优化**: 25_ampere_fprop_mainloop_fusion + 26_ampere_wgrad_mainloop_fusion
3. **稀疏利用**: 15_ampere_sparse_tensorop_gemm

#### Hopper调优路径
1. **Warp特化**: 48_hopper_warp_specialized_gemm
2. **FP8利用**: 54_hopper_fp8_warp_specialized_gemm
3. **TMA优化**: 50_hopper_gemm_with_epilogue_swizzle

#### Blackwell调优路径
1. **基础优化**: 70_blackwell_gemm
2. **Cluster调度**: 73_blackwell_gemm_preferred_cluster  
3. **Stream-K**: 74_blackwell_gemm_streamk

## 错误排查和调试

### 常见问题Examples
- **编译错误**: 从00_basic_gemm开始验证环境
- **运行时错误**: 使用02_dump_reg_smem分析内存
- **性能问题**: 参考47_ampere_gemm_universal_streamk并行策略
- **精度问题**: 参考27_ampere_3xtf32_fast_accurate_tensorop_gemm

### 调试工具链
```bash
# 基础调试
./02_dump_reg_smem/02_dump_reg_smem --help

# 布局分析  
./03_visualize_layout/03_visualize_layout --help

# 性能分析
./tools/profiler/cutlass_profiler --kernels=sgemm --verification-enabled=true
```

## 开发最佳实践

### 代码复用策略
1. **模板参数**: 参考48-61系列的参数化设计
2. **集合操作**: 使用49, 71的构建器模式
3. **融合模式**: 采用35, 37, 61的融合策略

### 测试验证流程
1. **功能验证**: 所有examples都包含验证逻辑
2. **性能验证**: 使用CUTLASS Profiler
3. **架构验证**: 在目标GPU上运行相应编号examples

### 文档和维护
- **代码注释**: 参考examples中的详细注释
- **参数说明**: 每个example都有完整的参数文档
- **版本兼容**: 注意examples与CUTLASS版本的兼容性要求