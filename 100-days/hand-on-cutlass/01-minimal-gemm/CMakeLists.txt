
# 启用 CUDA 支持
project(minimal_gemm LANGUAGES CXX CUDA)

# 查找 PyTorch（关键步骤）
find_package(Torch REQUIRED)

# 创建可执行文件或库
add_library(minimal_gemm SHARED minimal_gemm.cu)

# 链接 PyTorch（自动处理 include 路径）
target_link_libraries(minimal_gemm PRIVATE ${TORCH_LIBRARIES})

# 生成 compile_commands.json（供 IDE 使用）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)