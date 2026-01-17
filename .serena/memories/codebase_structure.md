# CUTLASS 代码库结构

## 顶级目录结构

```
cutlass/
├── include/                 # 头文件库 (客户端应用应包含此目录)
│   ├── cutlass/            # CUTLASS 核心模板库
│   └── cute/               # CuTe 布局、张量和 MMA/Copy 原子
├── examples/               # 92+ 个示例应用
├── tools/                  # 工具和实用程序
│   ├── library/           # CUTLASS 实例库
│   ├── profiler/          # 性能分析器
│   └── util/              # 实用程序
├── test/                   # 测试套件
│   ├── unit/              # GTest 单元测试
│   ├── python/            # Python 测试
│   └── self_contained_includes/ # 头文件自包含性检查
├── docs/                   # 文档
├── python/                 # Python 绑定和 CuTe DSL
├── cmake/                  # CMake 配置文件
├── media/                  # 媒体文件和图片
└── build/                  # 构建输出目录
```

## CUTLASS 核心库 (`include/cutlass/`)

### 主要模块
- **arch/** - 直接暴露架构特性 (包括指令级 GEMM)
- **gemm/** - 通用矩阵乘法专用代码
  - device/ - 设备级 GEMM 接口
  - kernel/ - 内核实现
  - threadblock/ - 线程块级组件
  - warp/ - warp 级组件
  - collective/ - 集合操作
- **conv/** - 卷积专用代码
- **epilogue/** - GEMM/卷积的 epilogue 代码
- **layout/** - 矩阵、张量内存布局定义
- **transform/** - 布局、类型、域转换专用代码
- **reduction/** - 带宽受限的 reduction 内核
- **thread/** - 在 CUDA 线程内执行的 SIMT 代码
- **platform/** - CUDA 兼容的标准库组件

### 核心头文件
- **cutlass.h** - 基础包含文件，状态码和常量
- **version.h** - 版本信息
- **numeric_types.h** - 数值类型定义
- **array.h** - 数组容器
- **matrix.h** - 矩阵抽象

## CuTe 库 (`include/cute/`)

### 核心组件
- **tensor.hpp** - 张量抽象
- **layout.hpp** - 布局代数
- **algorithm/** - 核心操作定义 (copy, gemm, tuple 操作)
- **arch/** - PTX 包装结构 (copy 和 math 指令)
- **atom/** - 元信息和原子操作
  - mma_atom.hpp - MMA 原子和 TiledMma
  - copy_atom.hpp - Copy 原子和 TiledCopy
  - *sm*.hpp - 特定架构的元信息
- **container/** - 容器类型
- **numeric/** - 数值操作
- **util/** - 实用程序

## 示例 (`examples/`)

### 示例分类
- **00-10**: 基础示例 (基本 GEMM, 实用程序, 布局)
- **11-20**: 复杂数据类型 (复数, quaternion)
- **21-30**: 高级 GEMM 变体 (分组, 融合)
- **31-40**: 特殊操作 (TRMM, SYRK, Python 接口)
- **41-50**: 注意力和高级融合
- **51-60**: Hopper 架构特性
- **61-70**: 混合精度和分组操作
- **71-80**: Blackwell 架构特性
- **81-92**: 最新特性 (稀疏, 分布式, MoE)

### 特殊目录
- **cute/** - CuTe 教程和示例
- **python/** - Python/CuTe DSL 示例
- **common/** - 共享实用程序和助手

## 测试结构 (`test/`)

### 单元测试组织
- **core/** - 核心功能测试
- **gemm/** - GEMM 操作测试
- **cute/** - CuTe 库测试
- **conv/** - 卷积测试
- **layout/** - 布局测试
- **util/** - 实用程序测试

### 测试类型
- 功能正确性测试
- 性能回归测试
- 架构特定测试
- 数据类型组合测试

## 工具 (`tools/`)

### 主要工具
- **profiler/** - 命令行性能分析工具
- **library/** - 实例化的 CUTLASS 模板库
- **util/** - 张量管理、参考实现、随机初始化实用程序

## 构建输出
- **build/** - CMake 构建目录
- 编译的二进制文件
- 生成的库文件
- 测试可执行文件