 Implementation Plan: Adding Bias Support to Hopper Grouped GEMM

 Overview

 Add per-row bias support to the Hopper grouped GEMM example (examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu) using CUTLASS's modern Epilogue
 Visitor Tree (EVT) framework.

 Current State Analysis

 Existing Implementation

 - File: examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu
 - Current Epilogue: LinearCombination<ElementC, ElementAccumulator> (line 152)
   - Computes: D = alpha * acc + beta * C
   - No bias support
 - Architecture: SM90 Hopper with TMA-based warp-specialized collective
 - Precision: FP8 input (E4M3 for A, E5M2 for B) → float accumulator → half_t output
 - Scheduling: Two modes (Cooperative and Pingpong)
 - Alpha/Beta Modes: Already supports scalar and per-group array modes (lines 569-592)

 Available Bias Solution

 - Fusion Operation: LinCombPerRowBias in include/cutlass/epilogue/fusion/operations.hpp (lines 151-166)
 - Formula: D = alpha * acc + beta * C + per-row bias
 - Callback: Sm90LinCombPerRowBias in include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp (lines 490-521)
 - Bias Stride: Stride<_1, _0, int64_t> = stride 1 in M dimension, 0 in N dimension, batch stride in 3rd dimension
 - Reference Examples:
   - examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu
   - examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu

 Implementation Strategy

 Design Decisions

 1. Bias Type: float (maintains precision through FP8→float→half_t pipeline)
 2. Bias Pattern: Per-row (N-dimensional vector, one per row, broadcast across columns)
 3. Memory Layout: Single contiguous allocation with concatenated per-group bias vectors
   - Group i bias stored at: bias_base + sum(N_0 to N_{i-1})
   - Batch stride: max(N_i) across all groups (conservative approach for variable N sizes)
 4. Runtime Control: --use_bias flag, nullable pointer for runtime disable

 Modification Steps

 Phase 1: Update Epilogue Fusion Operation (Line 152)

 Current:
 cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>

 New:
 cutlass::epilogue::fusion::LinCombPerRowBias<
   ElementC,              // ElementOutput = half_t
   ElementAccumulator,    // ElementCompute = float
   float,                 // ElementBias = float
   ElementC,              // ElementSource = half_t
   ElementAccumulator     // ElementScalar = float
 >

 Location: Lines 152 (inside both CooperativeConfig and PingpongConfig epilogue definitions)

 Phase 2: Add Bias Data Structures (Lines ~200-240)

 Add after block_D declaration:

 // Bias data structures
 using ElementBias = float;
 std::vector<ElementBias> bias_host;
 std::vector<int64_t> offset_bias(problem_count);
 cutlass::DeviceAllocation<ElementBias> block_bias;

 Phase 3: Add CLI Options (Lines ~244-384)

 Add in options parsing:

 bool use_bias = false;

 // In parse_options():
 if (arg == "--use_bias") {
   use_bias = true;
 }

 // In help text:
 std::cout << "  --use_bias                 Enable per-row bias addition\n";

 Phase 4: Allocate and Initialize Bias (Lines ~473-555)

 Add after C/D allocation:

 if (options.use_bias) {
   // Calculate max N for stride
   int max_N = 0;
   for (int32_t i = 0; i < problem_count; ++i) {
     max_N = std::max(max_N, problem_sizes_host[i].n());
   }

   // Calculate offsets and total size
   int64_t total_bias_elements = 0;
   for (int32_t i = 0; i < problem_count; ++i) {
     offset_bias[i] = total_bias_elements;
     total_bias_elements += problem_sizes_host[i].n();
   }

   // Allocate and initialize
   bias_host.resize(total_bias_elements);
   for (size_t i = 0; i < bias_host.size(); ++i) {
     bias_host[i] = ElementBias((rand() / double(RAND_MAX)) - 0.5);  // [-0.5, 0.5]
   }

   // Copy to device
   block_bias.reset(total_bias_elements);
   block_bias.copy_from_host(bias_host.data());
 }

 Phase 5: Configure Kernel Arguments (Lines ~557-614)

 Update fusion_args configuration:

 if (options.use_bias) {
   int max_N = 0;
   for (int32_t i = 0; i < problem_count; ++i) {
     max_N = std::max(max_N, problem_sizes_host[i].n());
   }

   fusion_args.bias_ptr = block_bias.get();
   fusion_args.dBias = {_1{}, _0{}, max_N};  // Stride<_1,_0,int64_t>
 } else {
   fusion_args.bias_ptr = nullptr;
   fusion_args.dBias = {_1{}, _0{}, 0};
 }

 Phase 6: Update Reference Computation (Lines ~680-730)

 Add bias to reference computation in verification loop:

 for (int m = 0; m < problem_m; ++m) {
   for (int n = 0; n < problem_n; ++n) {
     ElementC acc = ElementC(0);

     // GEMM computation
     for (int k = 0; k < problem_k; ++k) {
       int64_t idx_a = offset_A[problem_idx] + m * lda[problem_idx] + k;
       int64_t idx_b = offset_B[problem_idx] + k * ldb[problem_idx] + n;
       acc += ElementC(block_A.at(idx_a) * block_B.at(idx_b));
     }

     // Epilogue: D = alpha * acc + beta * C + bias
     int64_t idx_c = offset_C[problem_idx] + m * ldc[problem_idx] + n;
     int64_t idx_d = offset_D[problem_idx] + m * ldd[problem_idx] + n;
     ElementC c_value = block_C.at(idx_c);
     ElementC bias_value = ElementC(0);

     if (options.use_bias) {
       int64_t bias_idx = offset_bias[problem_idx] + n;  // Per-row: same bias for all m
       bias_value = ElementC(bias_host[bias_idx]);
     }

     block_D_reference.at(idx_d) = alpha_value * acc + beta_value * c_value + bias_value;
   }
 }

 Critical Files to Modify

 1. examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm.cu (primary implementation)
   - Line 152: Change fusion operation
   - Lines 200-240: Add bias data structures
   - Lines 244-384: Add CLI options
   - Lines 473-555: Allocate and initialize bias
   - Lines 557-614: Configure fusion_args
   - Lines 680-730: Update reference computation

 Reference Files (Read-Only)

 - include/cutlass/epilogue/fusion/operations.hpp (line 151-166): LinCombPerRowBias definition
 - include/cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp (lines 490-521): Callback arguments structure
 - examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu: Reference implementation

 Alternative Approaches

 Option 1: With Activation Function

 Replace LinCombPerRowBias with LinCombPerRowBiasEltAct<ReLU, ...> for fused bias + activation.

 Pros: Single kernel for bias + activation (common in neural networks)
 Cons: Less flexible, adds activation overhead

 Option 2: Per-Column Bias

 Use LinCombPerColBias for column-wise bias (M-dimensional vector).

 Pros: Useful for certain ML workloads
 Cons: Different memory pattern, less common

 Option 3: Pointer Array (Non-Contiguous)

 Store separate bias pointers for each group instead of concatenated allocation.

 Pros: More memory-efficient for highly variable N sizes
 Cons: Requires custom callback implementation, more complex

 Potential Issues and Mitigations

 1. Variable N Sizes in Grouped GEMM

 Issue: Different groups have different N dimensions, but stride must be uniform.
 Solution: Use max(N_i) as batch stride. Wastes address space but maintains correctness.

 2. Alignment Requirements

 Issue: AlignmentBias = 128 / sizeof_bits<float> = 4 elements.
 Mitigation: Most N sizes are multiples of 4 or larger. Add assertion for edge cases.

 3. Numerical Precision

 Issue: Accumulating FP8 → float → half_t, bias adds another rounding step.
 Mitigation: Use float for bias (not half_t) to minimize error accumulation.

 4. Performance Overhead

 Issue: Additional memory bandwidth for bias reads.
 Mitigation: Per-row bias reuses values across columns (good cache locality), TMA handles efficiently.

 Verification Strategy

 Unit Tests

 1. Single group, uniform N: Verify basic functionality
 2. Multiple groups, variable N: Test stride and offset calculation
 3. Edge cases: N=1, N not aligned, single element bias
 4. Both schedules: Cooperative and Pingpong modes
 5. Alpha/beta modes: Scalar and array modes with bias

 Validation Approach

 - Compare GPU output against CPU reference computation (with bias)
 - Check numerical error bounds (should be similar to current implementation)
 - Verify memory access patterns (use Nsight Compute)

 Performance Testing

 - Measure overhead with/without bias (expect 5-10%)
 - Profile with various problem sizes
 - Check TMA utilization remains high

 Success Criteria

 1. ✓ Code compiles without errors
 2. ✓ All tests pass with bias enabled
 3. ✓ Results match reference within tolerance (1e-3 for half_t)
 4. ✓ Runtime disable works (nullptr bias_ptr)
 5. ✓ Performance overhead < 15%
 6. ✓ Both Cooperative and Pingpong schedules work

 Estimated Complexity

 - Lines of Code: ~150 lines added
 - Implementation Time: Medium complexity
 - Risk Level: Low (additive changes, runtime disable fallback, well-documented pattern)