  核心关系概述

  在 CUTLASS 3.x 架构中，GemmKernel 和 Gemm 体现了设备端（Device-side）与主机端（Host-side）的分离设计：

  // 设备端：无状态的 GPU 核函数定义
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,           // 问题规模定义
      CollectiveMainloop,     // 主循环：负责矩阵乘法计算
      CollectiveEpilogue      // 尾声：负责输出和融合操作
  >;

  // 主机端：有状态的核函数适配器
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  详细架构分析

  1. GemmKernel（设备层）

  从 gemm_universal_decl.h:36-57 可以看到：

  /*
   * Stateless universal device GEMM kernel type that treats GEMM as
   * a composition of a collective mainloop and a collective epilogue.
   */
  template <
    class ProblemShapeOrThreadblockMma_,
    class CollectiveMainloopOrEpilogue_,
    class CollectiveEpilogueOrThreadblockSwizzle_,
    class TileScheduler_ = void,
    class Enable = void
  >
  class GemmUniversal;

  职责：
  - 无状态（Stateless）：纯粹的计算逻辑，不保存状态
  - 设备端执行：定义在 GPU 上运行的 __global__ 核函数
  - 组合式设计：将 GEMM 分解为两个核心组件
    - CollectiveMainloop：执行矩阵乘法的主循环（A×B 计算）
    - CollectiveEpilogue：执行输出操作（C = α·AB + β·C）

  在示例代码中的实例化：

  从 57_hopper_grouped_gemm.cu:155-164：

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutA *, AlignmentA,
      ElementB, LayoutB *, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<...>,
      KernelSchedule  // 例如：KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum
    >::CollectiveOp;

  这个 CollectiveMainloop 使用了 Hopper 架构的特性：
  - TMA (Tensor Memory Accelerator)：硬件加速的内存传输
  - GMMA (General Matrix Multiply-Accumulate)：硬件矩阵乘法单元
  - Warp-Specialized Cooperative：warp 级别的协作调度

  2. Gemm（主机层适配器）

  从 gemm_universal_adapter.h:68-82 可以看到：

  /*!
    GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
    of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

    It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
    to create it from the host facing arguments.
  */
  template <class GemmKernel_, class Enable = void>
  class GemmUniversalAdapter;

  职责：
  - 有状态（Stateful）：维护 Params params_ 成员变量（见 gemm_universal_adapter.h:221）
  - 主机端接口：提供用户友好的 C++ API
  - 生命周期管理：管理核函数参数的创建、初始化和销毁
  - 核函数启动：处理 CUDA 核函数的实际启动逻辑

  核心 API 方法：

  从 gemm_universal_adapter.h:230-616：

  // 1. 检查是否可以执行给定的问题
  static Status can_implement(Arguments const& args);

  // 2. 查询所需的工作空间大小
  static size_t get_workspace_size(Arguments const& args);

  // 3. 初始化核函数参数
  Status initialize(Arguments const& args, void* workspace, cudaStream_t stream);

  // 4. 执行核函数（静态方法，高级用户使用）
  static Status run(Params& params, cudaStream_t stream);

  // 5. 执行核函数（实例方法，常规使用）
  Status run(Arguments const& args, void* workspace, cudaStream_t stream);
  Status operator()(Arguments const& args, void* workspace, cudaStream_t stream);

  工作流程示例

  在 57_hopper_grouped_gemm.cu:658-723 中的 run() 函数展示了完整的使用流程：

  template <typename GemmT>
  int run(Options &options, bool host_problem_shapes_available = true) {
    // 1. 分配和初始化数据
    allocate(options);
    initialize(options);

    // 2. 实例化 Gemm 适配器（主机端对象）
    GemmT gemm;  // 这是 GemmUniversalAdapter<GemmKernel>

    // 3. 创建参数结构
    auto arguments = args_from_options<GemmT>(options, host_problem_shapes_available);

    // 4. 查询工作空间大小
    size_t workspace_size = GemmT::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // 5. 检查问题是否可以实现
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // 6. 初始化核函数（准备参数）
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // 7. 执行核函数
    CUTLASS_CHECK(gemm.run());  // 实际启动 GPU 核函数

    // 8. 验证结果
    result.passed = verify(options);
  }

  核函数启动的内部机制

  从 gemm_universal_adapter.h:372-575 可以看到 run() 方法的实现：

  static Status run(Params& params, cudaStream_t stream, ...) {
    // 1. 计算网格和线程块配置
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // 2.
