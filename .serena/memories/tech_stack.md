# CUTLASS 技术栈

## 编程语言
- **C++17**: 主要开发语言，要求 C++17 标准
- **CUDA C++**: GPU 内核开发，支持 CUDA 11.4+ (推荐 CUDA 12.8+)
- **Python**: CuTe DSL、工具和绑定，支持 Python 3.8+
- **PTX 汇编**: 低级 GPU 指令优化

## 构建系统
- **CMake**: 主要构建系统，最低版本要求 3.19
- **NVCC**: NVIDIA CUDA 编译器
- **支持的编译器**:
  - GCC 7.5+ (推荐 9.0+)
  - Clang 7.0+
  - MSVC (Windows, 当前有已知问题)

## 包管理
- **Pixi**: 现代包管理器，用于环境管理
- **setuptools**: Python 包构建
- **pip**: Python 包安装

## 依赖库

### CUDA 依赖
- **CUDA Toolkit**: 11.4-13.x
- **cuBLAS**: NVIDIA 基础线性代数子程序库
- **cuDNN**: 深度神经网络库 (可选)

### Python 依赖
- **cuda-python**: >=11.8.0 - CUDA Python 绑定
- **networkx**: 图处理
- **numpy**: 数值计算
- **pydot**: 图可视化
- **scipy**: 科学计算
- **treelib**: 树结构处理

### 测试框架
- **Google Test (GTest)**: C++ 单元测试
- **pytest**: Python 测试 (通过 CuTe DSL)

## 开发工具
- **Doxygen**: 文档生成
- **CUTLASS Profiler**: 性能分析工具
- **Python 代码生成器**: 自动生成内核代码

## 支持的平台
- **操作系统**: Linux (Ubuntu 18.04+), Windows (有限支持)
- **架构**: x86-64, aarch64
- **GPU 架构**: 
  - Volta (SM 7.0+)
  - Turing (SM 7.5)
  - Ampere (SM 8.0, 8.6, 8.9)
  - Hopper (SM 9.0)
  - Blackwell (SM 10.0, 12.0, 12.1)

## 特殊工具链要求
- **目标架构**: 某些功能需要 "a" 后缀 (如 sm_90a, sm_100a)
- **架构加速特性**: Hopper/Blackwell 特定指令需要正确的目标架构
- **向前兼容性**: PTX 代码通常向前兼容，但架构加速特性除外