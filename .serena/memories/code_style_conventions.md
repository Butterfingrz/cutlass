# CUTLASS 代码风格和约定

## 命名约定

### 类和结构体
- **PascalCase**: 类名使用大驼峰命名法
  - `Gemm`, `TensorRef`, `Layout`
  - 模板参数同样使用 PascalCase: `ElementA`, `LayoutA`

### 函数和方法
- **snake_case**: 函数名使用下划线分隔
  - `get_workspace_size()`, `can_implement()`, `crd2idx()`
- **特殊操作符**: `operator()` 用于函数对象

### 变量和成员
- **snake_case**: 变量名使用下划线分隔
  - `problem_size`, `ref_A`, `split_k_slices`
- **成员变量后缀**: 私有成员通常以下划线结尾
  - `underlying_operator_`

### 常量和枚举
- **kPascalCase**: 常量以 k 开头
  - `kSuccess`, `kErrorMisalignedOperand`, `kStages`
- **枚举值**: 通常以 k 开头
  - `kTensorwise`, `kBlockwise`

### 模板参数
- **明确的类型名**: 模板参数名清楚表达用途
  - `ElementA`, `ElementB`, `ElementC` (数据类型)
  - `LayoutA`, `LayoutB` (内存布局)
  - `ThreadblockShape`, `WarpShape` (块形状)

## 头文件组织

### 包含文件顺序
1. 对应的头文件 (如果是 .cpp 文件)
2. C/C++ 标准库头文件
3. 第三方库头文件
4. CUDA 相关头文件
5. CUTLASS 内部头文件

### 头文件保护
- 使用 `#pragma once` 而不是传统的头文件保护宏

### 前向声明
- 尽可能使用前向声明减少编译依赖

## 模板设计约定

### 模板参数列表
- **类型参数在前**: 数据类型参数列在前面
- **值参数在后**: 编译时常量参数列在后面
- **默认参数**: 提供合理的默认值

```cpp
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  int kStages = 3,
  int kAlignmentA = 128 / sizeof(ElementA),
  int kAlignmentB = 128 / sizeof(ElementB)
>
class Gemm;
```

### SFINAE 和概念
- 使用 SFINAE 进行模板约束
- 提供清晰的错误消息
- 使用 `enable_if_t` 等现代 C++ 特性

## 代码组织

### 命名空间
- **嵌套命名空间**: 反映目录结构
  - `cutlass::gemm::device`
  - `cutlass::layout`
  - `cute::algorithm`

### 文件命名
- **小写 + 下划线**: 文件名使用小写字母和下划线
  - `gemm.h`, `tensor_ref.h`, `mma_atom.hpp`
- **目录结构**: 反映命名空间结构

## 注释和文档

### 文件头注释
- 统一的版权声明
- BSD-3-Clause 许可证信息
- 文件描述

### 类和函数文档
- 使用 Doxygen 风格注释
- `\brief` 简要描述
- `\param` 参数说明
- `\return` 返回值说明

```cpp
/*! \file
    \brief Basic include for CUTLASS.
*/

/// Status code returned by CUTLASS operations
enum class Status {
  kSuccess,                ///< Operation completed successfully
  kErrorMisalignedOperand, ///< Operands are misaligned
  // ...
};
```

## 错误处理

### 状态码
- 使用 `cutlass::Status` 枚举返回操作状态
- 提供 `cutlassGetStatusString()` 获取错误描述

### 断言和检查
- 使用 CUDA 断言进行调试检查
- 编译时检查使用 `static_assert`

## CUDA 特定约定

### 设备函数标记
- `__device__`: 设备端函数
- `__host__ __device__`: 主机和设备端通用函数
- `__global__`: 内核函数

### 内存修饰符
- `__shared__`: 共享内存
- `__constant__`: 常量内存
- 明确指定内存空间

### 同步和原子操作
- 使用适当的同步原语
- 现代 CUDA 原子操作

## 性能约定

### 内联和强制内联
- 关键路径函数使用 `CUTLASS_DEVICE` 宏
- 避免不必要的函数调用开销

### 内存对齐
- 遵循 GPU 内存对齐要求
- 使用对齐的数据结构

### 寄存器和共享内存使用
- 优化寄存器使用
- 高效的共享内存访问模式

## 平台兼容性

### 编译器支持
- 支持 GCC, Clang, MSVC
- 使用标准 C++17 特性
- 避免编译器特定扩展

### CUDA 版本兼容性
- 支持 CUDA 11.4+
- 使用条件编译处理版本差异
- 架构特定代码隔离

## 测试约定

### 单元测试结构
- 每个主要组件都有对应测试
- 使用 Google Test 框架
- 测试文件命名: `test_[component].cpp`

### 测试组织
- 按功能模块组织测试
- 包含正确性和性能测试
- 支持不同架构的测试