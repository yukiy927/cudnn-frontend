#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"
#include <cudnn_frontend.h>
#include <cuda_runtime.h>

void fp8_benchmark_convolution(cudnn_frontend::DataType_t data_type, const std::string& name, int iterations) {
    namespace fe = cudnn_frontend;

    int64_t n = 1, c = 128, d = 9, h = 240, w = 360, k = 256, r = 3, s = 3, t = 3;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(data_type)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("image")
                               .set_dim({n, c, d, h, w})
                               .set_stride({c * d * h * w, 1, c * h * w, c * w, c})
                               .set_data_type(data_type));

    auto W = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("filter")
                               .set_dim({k, c, r, s, t})
                               .set_stride({c * r * s * t, 1, c * s * t, c * t, c})
                               .set_data_type(data_type));

    auto conv_options = fe::graph::Conv_fprop_attributes()
                            .set_padding({0, 0, 0})
                            .set_stride({1, 1, 1})
                            .set_dilation({1, 1, 1})
                            .set_name("conv");
    auto conv_output = graph->conv_fprop(X, W, conv_options);

    conv_output->set_output(true).set_data_type(data_type);

    REQUIRE(graph->validate().is_good());

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    REQUIRE(graph->build_operation_graph(handle).is_good());
    REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(graph->check_support(handle).is_good());
    REQUIRE(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    Surface<int8_t> X_gpu(n * c * d * h * w, false);
    Surface<int8_t> W_gpu(k * c * r * s * t, false);
    Surface<int8_t> Y_gpu(n * k * d * h * w, false);

    int64_t workspace_size;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_gpu.devPtr},
        {W, W_gpu.devPtr},
        {conv_output, Y_gpu.devPtr}};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        cudaEventRecord(start);
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }

    float average_time = total_time / iterations;
    std::cout << name << " convolution average time over " << iterations << " iterations: " << average_time << " ms" << std::endl;

    CUDNN_CHECK(cudnnDestroy(handle));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void bf16_benchmark_convolution(cudnn_frontend::DataType_t data_type, const std::string& name, int iterations) {
    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }

    int64_t n = 1, c = 128, d = 9, h = 240, w = 360, k = 256, r = 3, s = 3, t = 3;

    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, d, h, w})
                                   .set_stride({c * d * h * w, 1, c * h * w, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s, t})
                                   .set_stride({c * r * s * t, 1, c * s * t, c * t, c}));

        auto conv_options =
            fe::graph::Conv_fprop_attributes().set_padding({0, 0, 0}).set_stride({1, 1, 1}).set_dilation({1, 1, 1});
        auto Y = graph->conv_fprop(X, W, conv_options);

        Y->set_output(true);

        REQUIRE(graph->validate().is_good());

        REQUIRE(graph->build_operation_graph(handle).is_good());

        REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());

        REQUIRE(graph->check_support(handle).is_good());

        REQUIRE(graph->build_plans(handle).is_good());

        return std::make_tuple(graph, X, W, Y);
    };
    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    auto [graph, X, W, Y] = build_new_graph(handle);

    Surface<half> x_tensor(n * c * d * h * w, false);
    Surface<half> w_tensor(k * c * r * s * t, false);
    Surface<half> y_tensor(n * k * d * h * w, false);

    std::unordered_map<int64_t, void *> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    int64_t workspace_size;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph << std::endl;

    REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    
}


TEST_CASE("Benchmark FP8 vs BFLOAT16 convolution", "[benchmark][conv]") {
    int iterations = 10;
    fp8_benchmark_convolution(cudnn_frontend::DataType_t::FP8_E4M3, "FP8", iterations);
    bf16_benchmark_convolution(cudnn_frontend::DataType_t::BFLOAT16, "BFLOAT16", iterations);
    // benchmark_convolution(cudnn_frontend::DataType_t::BFLOAT16, "BFLOAT16", iterations);
}
