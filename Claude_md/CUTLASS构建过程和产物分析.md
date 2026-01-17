# CUTLASS Hopper-Release 构建过程与产物分析

## 概述

本文档详细分析 CUTLASS 项目使用 `cmake --preset hopper-release` 和 `cmake --build --preset hopper-release` 命令的构建过程、最终产物以及使用方法。

---

## 📊 当前构建状态

**从日志信息分析：**
```
[129/1685] Building CUDA object tools/library/CMakeFiles/
cutlass_library_gemm_sm90_bf16_gemm_bf16_objs.unity.09d7058f8d90.cu
```

- **构建进度：** 129/1685 (约 7.6%)
- **总编译目标：** 1685 个
- **当前状态：** 正在编译 SM90 (Hopper) 架构的 BF16 GEMM kernels
- **预计时间：** 构建仍需较长时间（取决于硬件，可能需要数小时）

---

## 🔧 第一阶段：cmake --preset hopper-release (配置阶段)

### 执行内容

当运行 `cmake --preset hopper-release` 时，CMake 读取 `CMakePresets.json` 并执行以下操作：

### 1. 读取预设配置 (CMakePresets.json:10-34)

```json
{
  "name": "hopper-release",
  "displayName": "Hopper (Release)",
  "description": "Optimized release build for Hopper architecture (SM90/SM90a) with Unity Build",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build/release"
}
```

### 2. 设置环境变量

```bash
PATH="${sourceDir}/.pixi/envs/default/bin:$PATH"
LD_LIBRARY_PATH="${sourceDir}/.pixi/envs/default/lib:$LD_LIBRARY_PATH"
CUDA_HOME="${sourceDir}/.pixi/envs/default"
```

### 3. 配置编译器路径 (来自 Pixi 环境)

| 编译器 | 路径 |
|--------|------|
| C 编译器 | `.pixi/envs/default/bin/gcc` |
| C++ 编译器 | `.pixi/envs/default/bin/g++` |
| CUDA 编译器 | `.pixi/envs/default/bin/nvcc` |
| 构建工具 | `.pixi/envs/default/bin/ninja` |

### 4. 设置 CUTLASS 构建选项

| 选项 | 值 | 说明 |
|------|-----|------|
| `CMAKE_BUILD_TYPE` | `Release` | 发布版本，最大优化 |
| `CUTLASS_NVCC_ARCHS` | `90a;90` | 目标架构：Hopper (SM90/SM90a) |
| `CUTLASS_UNITY_BUILD_ENABLED` | `ON` | Unity Build 加速编译 |
| `CUTLASS_ENABLE_LIBRARY` | `ON` | **编译 CUTLASS 库** |
| `CUTLASS_ENABLE_PROFILER` | `ON` | **编译性能分析工具** |
| `CUTLASS_ENABLE_TESTS` | `OFF` | 不编译测试 |
| `CUTLASS_ENABLE_EXAMPLES` | `OFF` | 不编译示例 |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | `ON` | 生成 compile_commands.json |

### 5. 生成构建文件

**输出文件：**
- `build/release/build.ninja` (5.1 MB) - Ninja 构建脚本
- `build/release/compile_commands.json` (1.4 MB) - 编译数据库（供 IDE 使用）
- `build/release/CMakeCache.txt` - CMake 缓存配置

---

## 🏗️ 第二阶段：cmake --build --preset hopper-release (构建阶段)

### 执行内容

运行 `cmake --build --preset hopper-release` 时，执行以下构建操作：

### 1. 构建配置

```json
{
  "name": "hopper-release",
  "configurePreset": "hopper-release",
  "jobs": 20  // 并行编译 20 个任务
}
```

### 2. 编译过程特点

**Unity Build 机制：**
- 多个 `.cu` 源文件合并成一个编译单元
- 文件命名模式：`*.unity.{hash}.cu`
- **优点：** 显著加速编译（减少编译器启动开销）
- **缺点：** 单个文件编译时间较长，内存占用大

**编译目标分类（1685 个目标）：**

| 分类 | 示例 | 数量估计 |
|------|------|---------|
| GEMM kernels | `gemm_sm90_bf16_gemm_bf16` | ~600 |
| Conv2D kernels | `conv2d_sm90_fprop_f16nhwc` | ~400 |
| Conv3D kernels | `conv3d_sm90_fprop_f16ndhwc` | ~100 |
| Sparse GEMM | `spgemm_e5m2_e4m3` | ~200 |
| Grouped GEMM | `gemm_grouped_bf16` | ~100 |
| 其他操作 | 工具和基础设施 | ~285 |

**支持的数据类型：**
- FP32, FP16, BF16, TF32
- INT8, INT4 (S4/U4), INT2 (S2/U2)
- FP8 (E4M3, E5M2) - Ada/Hopper 专用

**支持的架构：**
- SM50 (Maxwell), SM60 (Pascal), SM61
- SM75 (Turing), SM80 (Ampere)
- SM89 (Ada Lovelace)
- **SM90/SM90a (Hopper)** - 本次构建的主要目标

---

## 📦 最终编译产物

### 构建完成后的目录结构

```
build/release/
├── tools/
│   ├── library/
│   │   ├── libcutlass.so        # 共享库（主要产物）
│   │   └── libcutlass.a         # 静态库
│   └── profiler/
│       └── cutlass_profiler     # 性能分析工具（可执行文件）
├── include/
│   └── cutlass/                 # 生成的配置头文件
├── bin/                         # 可能包含其他工具
└── compile_commands.json        # 编译数据库
```

### 产物详解

#### 1️⃣ CUTLASS Library - 预编译 Kernel 库

**文件：** `libcutlass.so` (共享库) / `libcutlass.a` (静态库)

**配置来源：** `tools/library/CMakeLists.txt:38-39`
```cmake
option(CUTLASS_BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CUTLASS_BUILD_STATIC_LIBS "Build static libraries" ON)
```

**包含内容：**
- 1000+ 预编译的 GEMM/Conv kernels
- 多种数据类型和矩阵布局的组合
- 针对 Hopper (SM90) 优化的高性能实现
- 支持运行时 kernel 选择和调度

**库大小估计：** 共享库 ~500MB-1GB（取决于编译的 kernel 数量）

#### 2️⃣ CUTLASS Profiler - 性能分析工具

**文件：** `cutlass_profiler`

**功能：**
- Benchmark 不同 kernel 配置的性能
- 生成详细的性能报告（TFLOPS, 延迟, 带宽利用率）
- 帮助选择最优 kernel 配置
- 支持多种操作类型（GEMM, Conv, Sparse, Grouped）

**使用场景：**
- 性能调优和 kernel 选择
- 验证 kernel 正确性
- 生成性能基线数据

#### 3️⃣ Header Files - 开发接口

**位置：**
- 源代码头文件：`include/cutlass/` (项目根目录)
- 生成的头文件：`build/release/include/`

**用途：**
- Header-only 模式开发
- 链接库时的接口定义
- 自定义 kernel 实例化

---

## 🚀 编译产物使用方法

### 方法 1：链接预编译库（推荐 - 生产环境）

#### 适用场景
- 使用 CUTLASS 提供的标准 kernels
- 需要快速的应用程序编译时间
- 不需要自定义 kernel 配置

#### 安装步骤

**1. 等待构建完成**
```bash
# 监控构建进度
cmake --build --preset hopper-release

# 构建完成的标志：
# [1685/1685] Linking ...
```

**2. 安装到指定目录（推荐）**
```bash
cd build/release

# 安装到本地 install 目录
cmake -DCMAKE_INSTALL_PREFIX=./install .
cmake --build . --target install
```

**安装后的目录结构：**
```
install/
├── bin/
│   └── cutlass_profiler
├── include/
│   └── cutlass/
│       ├── gemm/
│       ├── conv/
│       └── ...
└── lib/
    ├── libcutlass.a
    ├── libcutlass.so
    └── cmake/
        └── cutlass/
            └── cutlass-config.cmake
```

#### 在项目中使用

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCutlassApp LANGUAGES CXX CUDA)

# 设置 CUTLASS 安装路径
set(CMAKE_PREFIX_PATH "/home/jovyan/cutlass/build/release/install")

# 查找 CUTLASS 包
find_package(cutlass REQUIRED)

# 创建可执行文件
add_executable(my_app main.cu)

# 链接 CUTLASS 库（自动处理头文件和库路径）
target_link_libraries(my_app PRIVATE cutlass::cutlass)
```

**main.cu 示例（使用库的运行时 API）:**
```cpp
#include <iostream>
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

int main() {
    // 使用运行时 API 查找 kernel
    cutlass::library::Handle handle;

    // Kernel 名称可以从 profiler 获取
    const char* kernel_name = "gemm_sm90_bf16_bf16_bf16_tensor_op_f32_128x128x32";

    auto* operation = handle.find_operation(kernel_name);
    if (!operation) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return -1;
    }

    std::cout << "Successfully found kernel: " << kernel_name << std::endl;

    // 配置参数并运行...
    // cutlass::library::GemmArguments args(...);
    // handle.run(operation, &args);

    return 0;
}
```

**编译和运行：**
```bash
mkdir build && cd build
cmake ..
make

# 设置库路径
export LD_LIBRARY_PATH=/home/jovyan/cutlass/build/release/install/lib:$LD_LIBRARY_PATH

# 运行
./my_app
```

**优点：**
- ✅ 应用程序编译速度快（无需编译 CUTLASS 模板）
- ✅ 可以使用所有预编译的 kernel
- ✅ 适合生产环境部署

**缺点：**
- ❌ 仅限于预编译的 kernel 配置
- ❌ 运行时 kernel 查找有小开销（通常可忽略）

---

### 方法 2：Header-Only 模式（最大灵活性）

#### 适用场景
- 需要自定义 kernel 配置
- 需要特殊的数据类型或布局组合
- 需要自定义 epilogue（输出处理）
- 研究和原型开发

#### 在项目中使用

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCutlassApp LANGUAGES CXX CUDA)

# 只需要头文件
include_directories(/home/jovyan/cutlass/include)
include_directories(/home/jovyan/cutlass/tools/util/include)

add_executable(my_app main.cu)

# 设置 CUDA 架构
set_target_properties(my_app PROPERTIES
    CUDA_ARCHITECTURES "90"  # Hopper
)
```

**main.cu 示例（直接模板实例化）:**
```cpp
#include <iostream>
#include "cutlass/gemm/device/gemm.h"

int main() {
    // 直接定义 GEMM 操作的模板参数
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::bfloat16_t,           // A 矩阵元素类型
        cutlass::layout::RowMajor,     // A 矩阵布局
        cutlass::bfloat16_t,           // B 矩阵元素类型
        cutlass::layout::RowMajor,     // B 矩阵布局
        cutlass::bfloat16_t,           // C 矩阵元素类型
        cutlass::layout::RowMajor,     // C 矩阵布局
        float,                         // 累加器类型
        cutlass::arch::OpClassTensorOp, // 使用 Tensor Core
        cutlass::arch::Sm90            // 目标架构
    >;

    // 实例化 kernel
    Gemm gemm_op;

    // 分配设备内存
    int M = 4096, N = 4096, K = 4096;
    cutlass::bfloat16_t *d_A, *d_B, *d_C, *d_D;
    // cudaMalloc(...);

    // 配置参数
    // Gemm::Arguments args(
    //     {M, N, K},           // 问题大小
    //     {d_A, K},            // A 矩阵和 leading dimension
    //     {d_B, N},            // B 矩阵和 leading dimension
    //     {d_C, N},            // C 矩阵和 leading dimension
    //     {d_D, N},            // D 矩阵和 leading dimension
    //     {1.0f, 0.0f}         // alpha, beta
    // );

    // 运行 kernel
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // cutlass::Status status = gemm_op(args, nullptr, stream);

    std::cout << "Header-only GEMM instantiated successfully!" << std::endl;

    return 0;
}
```

**编译：**
```bash
nvcc main.cu -o my_app \
  -I/home/jovyan/cutlass/include \
  -I/home/jovyan/cutlass/tools/util/include \
  -arch=sm_90 \
  -std=c++17 \
  -O3
```

**优点：**
- ✅ 完全自定义 kernel 配置
- ✅ 编译器可以针对特定用例优化
- ✅ 无运行时查找开销
- ✅ 适合研究和原型开发

**缺点：**
- ❌ 应用程序编译时间极长（可能数十分钟）
- ❌ 需要 CUDA 编译器
- ❌ 二进制文件可能很大

---

### 方法 3：使用 CUTLASS Profiler（性能分析）

#### 主要用途
- 发现最快的 kernel 配置
- 生成性能基准数据
- 验证 kernel 正确性
- 为方法 1 选择最优 kernel

#### 基本使用

**1. 等待构建完成并找到可执行文件：**
```bash
# Profiler 位置
ls -lh build/release/tools/profiler/cutlass_profiler
# 或
ls -lh build/release/install/bin/cutlass_profiler
```

**2. 列出可用的 kernels：**
```bash
cd build/release/tools/profiler

# 列出所有 BF16 GEMM kernels
./cutlass_profiler --kernels=gemm --op_class=tensorop --accum=f32 --element=bf16

# 输出会显示所有可用的 kernel 名称
```

**3. 性能测试：**

**测试单个问题大小：**
```bash
./cutlass_profiler \
  --kernels=gemm \
  --m=4096 --n=4096 --k=4096 \
  --op_class=tensorop \
  --accum=f32 \
  --element=bf16 \
  --warmup-iterations=10 \
  --profiling-iterations=100
```

**测试多个问题大小（扫描）：**
```bash
./cutlass_profiler \
  --kernels=gemm \
  --m=1024:8192:1024 \
  --n=1024:8192:1024 \
  --k=1024:8192:1024 \
  --op_class=tensorop \
  --element=bf16
```

**4. 输出解析：**

典型输出格式：
```
Operation,Provider,Problem,Arguments,ElementA,ElementB,ElementC,ElementAccum,...,Runtime(ms),GFLOPS
gemm_sm90_bf16_...,cutlass,4096x4096x4096,...,bf16,bf16,bf16,f32,...,0.345,250123.4
gemm_sm90_bf16_...,cutlass,4096x4096x4096,...,bf16,bf16,bf16,f32,...,0.352,245678.2
...
```

**关键列：**
- `Operation`: Kernel 名称（用于方法 1 的运行时 API）
- `Runtime(ms)`: 平均运行时间
- `GFLOPS` 或 `TFLOPS`: 吞吐量（越高越好）

**5. 保存结果：**
```bash
./cutlass_profiler --kernels=gemm ... > results.csv
```

#### 高级用法

**测试 Convolution：**
```bash
./cutlass_profiler \
  --kernels=conv2d \
  --n=1 --h=224 --w=224 --c=64 --k=64 \
  --r=3 --s=3 \
  --pad_h=1 --pad_w=1
```

**测试 Sparse GEMM：**
```bash
./cutlass_profiler \
  --kernels=spgemm \
  --m=4096 --n=4096 --k=4096 \
  --sparsity=0.5
```

**指定特定 kernel：**
```bash
./cutlass_profiler \
  --operation=gemm_sm90_bf16_bf16_bf16_tensor_op_f32_128x128x32_3x_align16
```

---

## 📊 构建进度监控

### 检查构建是否完成

**方法 1：查看 Ninja 输出**
```bash
cmake --build --preset hopper-release

# 完成时会显示：
# [1685/1685] Linking CXX shared library ...
```

**方法 2：检查库文件是否生成**
```bash
# 查找共享库
find build/release -name "libcutlass.so*"

# 查找静态库
find build/release -name "libcutlass.a"

# 查找 profiler
find build/release -name "cutlass_profiler" -type f
```

**方法 3：查看构建日志**
```bash
# 将输出保存到文件
cmake --build --preset hopper-release 2>&1 | tee build.log

# 查看最后几行
tail -f build.log
```

### 估算剩余时间

根据当前进度 (129/1685 = 7.6%)，如果：
- 单个编译单元平均耗时：30-60 秒（取决于硬件）
- 并行任务数：20 (从 buildPresets.jobs)

**粗略估算：**
- 剩余目标：1685 - 129 = 1556
- 并行批次：1556 / 20 ≈ 78 批
- 总时间：78 × 45秒 ≈ 58 分钟（理想情况）
- 实际时间：1-3 小时（考虑系统负载和依赖关系）

---

## 🔍 常见问题排查

### Q1: 编译失败，报 CUDA 错误

**可能原因：**
- GPU 架构不匹配
- CUDA 版本不兼容
- 内存不足

**解决方法：**
```bash
# 检查 GPU 架构
nvidia-smi --query-gpu=compute_cap --format=csv

# 如果不是 Hopper (9.0)，修改 CMakePresets.json:
"CUTLASS_NVCC_ARCHS": "89"  # 改为你的架构

# 检查 CUDA 版本
nvcc --version

# 释放内存，减少并行任务
cmake --build --preset hopper-release -- -j 10
```

### Q2: 编译速度很慢

**原因：** Unity Build 将大量代码合并，单个文件编译时间长

**优化方法：**
```bash
# 增加并行任务数（如果内存足够）
cmake --build --preset hopper-release -- -j 30

# 或关闭 Unity Build（会更慢，但更稳定）
cmake --preset hopper-release -DCUTLASS_UNITY_BUILD_ENABLED=OFF
cmake --build --preset hopper-release
```

### Q3: 链接时找不到 libcutlass.so

**解决方法：**
```bash
# 设置库路径
export LD_LIBRARY_PATH=/home/jovyan/cutlass/build/release/tools/library:$LD_LIBRARY_PATH

# 或在 CMakeLists.txt 中设置 RPATH
set_target_properties(my_app PROPERTIES
    INSTALL_RPATH "/path/to/cutlass/lib"
)
```

### Q4: Profiler 提示 "Operation not found"

**原因：** 请求的 kernel 配置没有编译

**解决方法：**
```bash
# 列出实际可用的 kernels
./cutlass_profiler --kernels=gemm --op_class=tensorop

# 或使用通配符测试所有 kernels
./cutlass_profiler --operation=* --m=1024 --n=1024 --k=1024
```

---

## 📚 推荐工作流

### 新手入门流程

1. **等待构建完成**
   ```bash
   cmake --build --preset hopper-release
   # 去喝杯咖啡☕，回来查看是否完成
   ```

2. **安装到本地目录**
   ```bash
   cd build/release
   cmake -DCMAKE_INSTALL_PREFIX=./install .
   cmake --build . --target install
   ```

3. **使用 Profiler 探索**
   ```bash
   cd install/bin
   ./cutlass_profiler --kernels=gemm --m=2048 --n=2048 --k=2048
   ```

4. **在简单项目中测试**
   - 使用方法 1（链接库）创建一个 Hello World 程序
   - 验证能正确调用 CUTLASS

5. **根据需求选择使用方式**
   - 生产环境 → 方法 1（链接库）
   - 研究开发 → 方法 2（Header-only）
   - 性能调优 → 方法 3（Profiler）

### 高级开发流程

1. **使用 Profiler 确定最优 kernel**
2. **在生产代码中使用运行时 API 调用**
3. **对于特殊需求，使用 Header-only 模式自定义**
4. **定期使用 Profiler 验证性能**

---

## 📖 参考资源

- **CUTLASS 官方文档：** https://github.com/NVIDIA/cutlass/tree/main/media/docs
- **Profiler 文档：** https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiler.md
- **示例代码：** `examples/` 目录
- **CMake 配置：** `CMakePresets.json`, `CMakeLists.txt`

---

**文档生成时间：** 2025-11-09
**分析工具：** Claude Code + Zen MCP Thinkdeep + Expert Analysis
**项目：** NVIDIA CUTLASS
**构建配置：** hopper-release (SM90/SM90a, Release, Unity Build)
