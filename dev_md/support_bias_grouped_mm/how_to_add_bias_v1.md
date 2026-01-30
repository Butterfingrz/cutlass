###  一、57示例现在的结构（你要改哪里）

    - examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu:559 的 args_from_options() 用
    decltype(arguments.epilogue.thread) fusion_args;
    来构造 epilogue 侧的运行时参数，目前只填了 alpha/beta 及其“标量/每组指针数组”两种模式（见 :569-592）。


#### 1.1、关键是：CUTLASS里“PerRow/PerCol”指的是 对输出矩阵 D(M×N) 的哪个维度变化：

  - LinCombPerRowBias：bias 向量长度是 M（每一行一个bias，沿N广播）
  - LinCombPerColBias：bias 向量长度是 N（每一列一个bias，沿M广播）


### 二、Bias在Hopper( SM90 ) epilogue(EVT)里是怎么“接进去”的

  参考你给的文档：dev_md/grouped_gemm/Bias_Support_to_Hopper_grouped_gemm.md，它的主线是正确的：用 CUTLASS 3 的 EVT（Epilogue Visitor Tree）把
  bias 融合进 epilogue，而不是再写一个额外kernel。

  关键是：CUTLASS里“PerRow/PerCol”指的是 对输出矩阵 D(M×N) 的哪个维度变化：

  - LinCombPerRowBias：bias 向量长度是 M（每一行一个bias，沿N广播）
  - LinCombPerColBias：bias 向量长度是 N（每一列一个bias，沿M广播）

#### 2.1、这里应该需要沿着`M`维度的 `bias`，在 cutlass 中称为 `per_row` (`容易搞错`)


  这一点可以从实现直接看出来：

  - include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp 里
    LinCombPerRowBias 用的是 Sm90ColBroadcast<... Stride<_1,_0,int64_t> ...>（bias随M变、N维stride为0）
  - 同文件里 LinCombPerColBias 用的是 Sm90RowBroadcast<... Stride<_0,_1,int64_t> ...>（bias随N变、M维stride为0）

  并且，Bias相关的运行时参数结构已经由FusionCallbacks定义好了（你只需要填）：

  - bias_ptr：bias基址（device指针）
  - dBias：Stride<_1,_0,int64_t> 或 Stride<_0,_1,int64_t>，第三维 int64_t 是 group维(L维)的batch stride


### 三、修改策略

  #### 3.1 先把GEMM从FP8改成BF16（否则schedule/指令不匹配）

  57现在是：

  - ElementA = float_e4m3_t、ElementB = float_e5m2_t、ElementC = half_t（见 examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu:94-96）
  - kernel schedule 是 KernelPtrArrayTmaWarpSpecialized*FP8FastAccum（见 :124、:131）

  要做“bf16精度的grouped_gemm”，通常至少要做到：

  - A/B 为 cutlass::bfloat16_t
  - C/D 为 cutlass::bfloat16_t（如果你要输出bf16）
  - ElementAccumulator 保持 float（目前就是 float，见 :117，这对bf16是常规/推荐）

  同时，kernel schedule 必须换成非FP8的通用版本，否则编译期就会因为schedule与数据类型组合不支持而失败。可以直接参考半精度ptr-array例子：

  - 参考：examples/56_hopper_ptr_array_batched_gemm/56_hopper_ptr_array_batched_gemm.cu:88-121

  建议改法（概念上）：

  - CooperativeConfig::KernelSchedule：从 KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum 换成
    KernelPtrArrayTmaWarpSpecializedCooperative
  - PingpongConfig::KernelSchedule：从 ...PingpongFP8FastAccum 换成 ...Pingpong
  - TileShape 的 K 建议跟着 half/bf16 的常用配置走（例如示例56用 _64），否则可能触发builder选择不到合适GMMA管线。你可以先用示例56那套tile/
    cluster作为起点，再做性能调参。

  ———

  #### 3.2 Epilogue：从LinearCombination换成带Bias的fusion op

  现在的epilogue是（examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu:152）：

  - cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>

  要支持bias，你换成以下二选一（取决于你的bias定义）：

  A) 典型NN Bias（长度N，每列一个bias，沿M广播）

  - 用：cutlass::epilogue::fusion::LinCombPerColBias<...>
  - bias索引语义：bias[n]（对所有m共用）

  B) 真“每行一个bias”（长度M，每行一个bias，沿N广播）

  - 用：cutlass::epilogue::fusion::LinCombPerRowBias<...>
  - bias索引语义：bias[m]（对所有n共用）

  模板参数推荐写法（以 BF16 输出、float compute 为例）：

  - ElementOutput = ElementC（bf16）
  - ElementCompute = ElementAccumulator（float）
  - ElementBias：
      - 想严格“bf16精度链路”：用 cutlass::bfloat16_t
      - 想更高数值精度（常见做法）：用 float（bias存float，epilogue里转float再算，最后cast到bf16）

  ———

  #### 3.3 Bias内存布局：Grouped场景下必须解决“每组尺寸不同但stride必须统一”

  这是你给的文档里最容易踩坑的地方：EVT默认的 bias_ptr + l * dBias[2] 只能表达“固定stride的批次布局”。

  也就是说：

  - epilogue不会接受“每组一个不同的offset_bias[i]”（除非你自己做 pointer-array bias 的自定义visitor/callback）
  - 所以最稳的方案是：为bias做padding，令每个group的bias起始地址满足固定间隔

  具体做法（按你选的PerRow/PerCol分别）：

  - 若用 LinCombPerColBias（bias长度N）
      - 计算 strideL = round_up(max_N_across_groups, AlignmentBias)
      - 分配 bias_device 大小 groups * strideL
      - 第 g 组的bias向量起点：bias_ptr + g*strideL，有效范围 [0, N_g)
      - dBias = { _0{}, _1{}, strideL }（注意：这是 Stride<_0,_1,int64_t>）
  - 若用 LinCombPerRowBias（bias长度M）
      - 计算 strideL = round_up(max_M_across_groups, AlignmentBias)
      - 分配 bias_device 大小 groups * strideL
      - 第 g 组的bias向量起点：bias_ptr + g*strideL，有效范围 [0, M_g)
      - dBias = { _1{}, _0{}, strideL }（这是 Stride<_1,_0,int64_t>）

  为什么要 round_up(..., AlignmentBias)：bias broadcast内部会做向量化 V = min(AlignmentBias, ...)，让每组起点保持对齐能避免潜在的非对齐向量load问题，同时性能更稳。