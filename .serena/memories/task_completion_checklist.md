# CUTLASS 任务完成检查清单

## 代码修改后的必要步骤

### 1. 编译检查
```bash
# 基础编译测试
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a;100a"
make -j$(nproc)
```

### 2. 单元测试
```bash
# 运行相关单元测试
make test_unit -j
# 或运行特定测试
./test/unit/gemm/device/cutlass_test_unit_gemm_device
./test/unit/cute/cute_test_unit
```

### 3. 示例验证
```bash
# 编译并运行相关示例
make examples -j
# 运行修改涉及的示例
./examples/00_basic_gemm/00_basic_gemm
```

### 4. 性能回归检查
```bash
# 使用 profiler 进行性能检查
make cutlass_profiler -j16
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=1024
```

### 5. 代码风格检查
- 检查命名约定是否符合项目标准
- 确保头文件包含顺序正确
- 验证注释和文档完整性
- 检查模板参数命名

### 6. 架构兼容性验证
```bash
# 多架构编译测试
cmake .. -DCUTLASS_NVCC_ARCHS="75;80;86;89;90;90a;100;100a;120;120a"
make -j
```

### 7. Python 绑定测试 (如果涉及)
```bash
cd python
python -m pytest test/
python -m pytest cutlass_library/test/
```

## 特定修改类型的检查

### GEMM 内核修改
- [ ] 编译 GEMM 相关示例
- [ ] 运行 GEMM 单元测试
- [ ] 性能基准测试
- [ ] 不同数据类型组合测试
- [ ] 不同矩阵大小测试

### CuTe 库修改
- [ ] CuTe 单元测试
- [ ] 布局和张量操作测试
- [ ] CuTe DSL Python 测试
- [ ] CuTe 示例验证

### 新架构支持
- [ ] 架构特定编译测试
- [ ] 新指令验证
- [ ] 性能优化验证
- [ ] 向后兼容性检查

### 数据类型支持
- [ ] 数值精度测试
- [ ] 转换操作测试
- [ ] 边界条件测试
- [ ] 性能影响评估

## 文档更新检查

### 必要文档更新
- [ ] README.md (如有 API 变更)
- [ ] CHANGELOG.md (记录变更)
- [ ] 示例文档更新
- [ ] API 文档更新 (Doxygen)

### 版本控制检查
- [ ] 提交消息清晰描述变更
- [ ] 相关 issue 引用
- [ ] 分支命名符合约定
- [ ] 代码审查完成

## 性能验证

### 基准测试
```bash
# 运行性能基准
./tools/profiler/cutlass_profiler --kernels=cutlass_tensorop_s* --m=4096 --n=4096 --k=4096
```

### 内存使用检查
```bash
# 使用 compute-sanitizer 检查内存
compute-sanitizer --tool=memcheck ./your_program
```

### 正确性验证
```bash
# 与参考实现比较
./tools/profiler/cutlass_profiler --verification-enabled=true
```

## 发布前检查

### 集成测试
- [ ] 完整测试套件运行
- [ ] 多个 GPU 架构测试
- [ ] 不同 CUDA 版本测试
- [ ] 性能回归测试

### 依赖检查
- [ ] 最小依赖版本验证
- [ ] 新增依赖必要性评估
- [ ] 许可证兼容性检查

### 文档完整性
- [ ] 新功能使用示例
- [ ] API 变更说明
- [ ] 迁移指南 (如需要)
- [ ] 性能特征说明

## 常见问题排查

### 编译失败
1. 检查 CUDA 版本兼容性
2. 验证编译器版本
3. 检查 CMake 配置
4. 确认架构目标设置

### 测试失败
1. 检查 GPU 架构支持
2. 验证输入参数有效性
3. 检查内存对齐要求
4. 确认数值精度设置

### 性能问题
1. 验证最优编译选项
2. 检查内存访问模式
3. 分析占用率和吞吐量
4. 与基线性能比较

## 自动化检查脚本示例

```bash
#!/bin/bash
# 完整验证脚本

set -e

echo "=== CUTLASS 任务完成验证 ==="

echo "1. 清理并重新构建..."
rm -rf build && mkdir build && cd build

echo "2. 配置构建..."
cmake .. -DCUTLASS_NVCC_ARCHS="80;90a"

echo "3. 编译..."
make -j$(nproc)

echo "4. 运行单元测试..."
make test_unit -j

echo "5. 编译示例..."
make examples -j

echo "6. 运行基础示例..."
./examples/00_basic_gemm/00_basic_gemm

echo "7. 性能测试..."
./tools/profiler/cutlass_profiler --kernels=sgemm --m=1024 --n=1024 --k=1024

echo "=== 验证完成 ==="
```