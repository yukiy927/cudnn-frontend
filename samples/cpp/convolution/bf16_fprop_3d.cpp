#include <catch2/catch_test_macros.hpp>
#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Convolution fprop", "[conv][graph][caching]") {
    namespace fe = cudnn_frontend;

    if (is_arch_supported_by_cudnn() == false) {
        SKIP("Architecture is not supported by currend cudnn version");
    }

    int64_t n = 1, c = 128, d = 9, h = 14, w = 12; // c = input channel
    int64_t k = 128;                // k = output channel
    int64_t r = 3, s = 3, t = 3;   // (r, s, t) = kernel_size

    // Calculate output dimensions
    int64_t out_d = (d - r + 1);
    int64_t out_h = (h - s + 1);
    int64_t out_w = (w - t + 1);
    
    auto build_new_graph = [=](cudnnHandle_t handle) {
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
                .set_compute_data_type(fe::DataType_t::FLOAT);

        auto X = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("image")
                                   .set_dim({n, c, d, h, w})
                                   .set_stride({c * d * h * w, 1, c * h * w, c * w, c}));

        auto W = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("filter")
                                   .set_dim({k, c, r, s, t})
                                   .set_stride({c * r * s * t, 1, c * s * t, c * t, c}));

        auto conv_options = fe::graph::Conv_fprop_attributes()
                                .set_padding({0, 0, 0})
                                .set_stride({1, 1, 1})
                                .set_dilation({1, 1, 1});
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
    Surface<half> y_tensor(n * k * out_d * out_h * out_w, false);

    std::unordered_map<int64_t, void *> variant_pack = {
        {X->get_uid(), x_tensor.devPtr}, {W->get_uid(), w_tensor.devPtr}, {Y->get_uid(), y_tensor.devPtr}};

    int64_t workspace_size;
    REQUIRE(graph->get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::cout << *graph << std::endl;

    for (int i = 0; i < 10; ++i) {
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
    }

}