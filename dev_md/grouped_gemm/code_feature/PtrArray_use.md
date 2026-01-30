 ## 1) “PtrArray” 在这个 grouped_gemm 里的真实含义

  这里的 grouped GEMM 不是把所有 group 打包成一个大张量再做 strided-batched；而是走 指针数组接口：

  - host/device 侧为每个 group 准备：
      - A_ptr[group] / B_ptr[group] / C_ptr[group] / D_ptr[group]
      - strideA[group] / strideB[group] / strideC[group] / strideD[group]
      - problem_shape[group] = (M,N,K)（本例 ProblemShape 是 GroupProblemShape<Shape<int,int,int>>）
  - 你在构造 arguments 时就能看到：mainloop 参数只给了 ptr_A + stride_A + ptr_B + stride_B，epilogue 参数给了 ptr_C + stride_C + ptr_D + stride_D。见
    examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu:595。

  为什么 schedule 名里要强调 PtrArray？

  - 因为 TMA 需要 TensorMap descriptor 描述“从 gmem 读取一个张量 tile 的维度/步长/基址”，而 ptr-array/grouped 会让 基址、stride、甚至 (M,N,K) 都随
    group 变化，所以必须有一套“按 group 切换 descriptor”的机制。