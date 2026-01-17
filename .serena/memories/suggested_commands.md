# CUTLASS 开发常用命令

## 构建命令

### 基础构建
```bash
# 设置 CUDA 编译器环境变量
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc

# 创建构建目录
mkdir build && cd build

# 基础 CMake 配置
cmake .. -DCUTLASS_NVCC_ARCHS=80  # 针对 Ampere 架构

# 编译
make -j$(nproc)
```

### 架构特定构建
```bash
# Hopper 架构 (需要 "a" 后缀用于架构加速特性)
cmake .. -DCUTLASS_NVCC_ARCHS="90a"

# Blackwell 架构
cmake .. -DCUTLASS_NVCC_ARCHS="100a"

# 多架构构建
cmake .. -DCUTLASS_NVCC_ARCHS="75;80;90a"
```

### 特殊构建配置
```bash
# 仅头文件库
cmake .. -DCUTLASS_ENABLE_HEADERS_ONLY=ON

# 构建所有内核 (警告: 构建时间很长)
cmake .. -DCUTLASS_LIBRARY_KERNELS=all

# 构建特定内核子集
cmake .. -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
```

## 测试命令

### 单元测试
```bash
# 构建并运行所有单元测试
make test_unit -j

# 运行特定测试类别
./test/unit/gemm/device/cutlass_test_unit_gemm_device
./test/unit/cute/cute_test_unit

# 使用 CTest 运行测试
ctest --output-on-failure
```

### Python 测试
```bash
# 进入 Python 目录
cd python

# 运行 Python 测试
python -m pytest test/

# 运行 CuTe DSL 测试
python -m pytest cutlass_library/test/
```

## 性能分析

### CUTLASS Profiler
```bash
# 构建 profiler
make cutlass_profiler -j16

# 基础性能分析
./tools/profiler/cutlass_profiler --kernels=sgemm --m=3456 --n=4096 --k=4096

# 分析特定内核类型
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align8 --m=3456 --n=4096 --k=4096

# 卷积分析
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s*fprop_optimized_f16 --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3
```

## 包管理命令

### Pixi 环境管理
```bash
# 安装依赖
pixi install

# 激活环境
pixi shell

# 运行特定任务
pixi run [task_name]
```

### Python 包管理
```bash
# 安装 CUTLASS Python 包
pip install nvidia-cutlass

# 开发模式安装
pip install -e .

# 安装依赖
pip install -r requirements.txt
```

## 开发工具命令

### 文档生成
```bash
# 生成 Doxygen 文档 (如果 Doxygen 可用)
make docs

# 查看在线文档
# https://docs.nvidia.com/cutlass/
```

### 代码格式化和检查
```bash
# 目前没有统一的代码格式化工具
# 建议遵循现有代码风格
```

## 示例运行

### 编译和运行示例
```bash
# 构建所有示例
make examples -j

# 运行特定示例
./examples/00_basic_gemm/00_basic_gemm
./examples/cute/tutorial/blackwell/00_hello_world

# Python/CuTe DSL 示例
cd examples/python/CuTeDSL
python examples/ampere/basic_gemm.py
```

## 环境检查命令

### 系统信息
```bash
# 检查 CUDA 版本
nvcc --version
nvidia-smi

# 检查 CMake 版本
cmake --version

# 检查编译器版本
gcc --version
clang --version
```

### 依赖检查
```bash
# 检查 Python 环境
python --version
pip list | grep -E "(cuda|numpy|scipy)"

# 检查 GPU 信息
nvidia-smi -q
```

## 清理命令

### 构建清理
```bash
# 清理构建目录
rm -rf build/

# 仅清理编译产物
make clean

# Git 清理未跟踪文件
git clean -fd
```

## 调试命令

### 构建调试
```bash
# Debug 构建
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 详细构建输出
make VERBOSE=1

# 并行度控制 (避免内存不足)
make -j4  # 限制并行作业数
```

### 运行时调试
```bash
# 使用 cuda-gdb 调试
cuda-gdb ./your_program

# 使用 compute-sanitizer 检查
compute-sanitizer ./your_program
```