/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

 #include <catch2/catch_test_macros.hpp>
 #include "../utils/helpers.h"
 
 #include <cudnn_frontend.h>
 
 TEST_CASE("Convolution fp8 precision", "[conv][graph]") {
     if (cudnnGetCudartVersion() < 12000) {
         SKIP("Test requires cuda toolkit 12.0 or above");
     }
     if (cudnnGetVersion() < 8600) {
         SKIP("TEST REQUIRES minimum cudnn version 8.6.0");
     }
     if (check_device_arch_newer_than("hopper") == false) {
         SKIP("TEST REQUIRES device  hopper arch or newer");
     }
 
     namespace fe = cudnn_frontend;

     int64_t n = 1, c = 256, d = 3, h = 14, w = 12; // c = input channel
     int64_t k = 512;                // k = output channel
     int64_t r = 3, s = 3, t = 3;   // (r, s, t) = kernel_size
     
     // Calculate output dimensions
     int64_t out_d = (d - r + 1);
     int64_t out_h = (h - s + 1);
     int64_t out_w = (w - t + 1);
 
     // Initialize input tensors with int8_t as proxy for fp8
     auto graph = std::make_shared<fe::graph::Graph>();
     graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
 
     auto X = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("image")
                                .set_dim({n, c, d, h, w})
                                .set_stride({c * d * h * w, 1, c * h * w, c * w, c})
                                .set_data_type(fe::DataType_t::FP8_E4M3));
 
     auto W = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("filter")
                                .set_dim({k, c, r, s, t})
                                .set_stride({c * r * s * t, 1, c * s * t, c * t, c})
                                .set_data_type(fe::DataType_t::FP8_E4M3));
 
     auto conv_options = fe::graph::Conv_fprop_attributes()
                             .set_padding({0, 0, 0})
                             .set_stride({1, 1, 1})
                             .set_dilation({1, 1, 1})
                             .set_name("conv");
     auto conv_output_fp8 = graph->conv_fprop(X, W, conv_options);
 
     auto descale_x = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("descale_x")
                                        .set_dim({1, 1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::FLOAT));
 
     auto descale_w = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("descale_w")
                                        .set_dim({1, 1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::FLOAT));
 
     auto scale_y = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("scale_y")
                                      .set_dim({1, 1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::FLOAT));
 
     auto scale_options   = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
     auto after_descale_x = graph->pointwise(conv_output_fp8, descale_x, scale_options);
     auto after_descale_w = graph->pointwise(after_descale_x, descale_w, scale_options);
     auto Y               = graph->pointwise(after_descale_w, scale_y, scale_options);
 
     Y->set_output(true).set_data_type(fe::DataType_t::FP8_E4M3);
 
     auto amax = graph->reduction(after_descale_w,
                                  fe::graph::Reduction_attributes()
                                      .set_mode(fe::ReductionMode_t::AMAX)
                                      .set_compute_data_type(fe::DataType_t::FLOAT));
 
     amax->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, 1, 1, 1, 1});
 
     REQUIRE(graph->validate().is_good());
 
     cudnnHandle_t handle;
     CUDNN_CHECK(cudnnCreate(&handle));
 
     REQUIRE(graph->build_operation_graph(handle).is_good());
     REQUIRE(graph->create_execution_plans({fe::HeurMode_t::A}).is_good());
 
     REQUIRE(graph->check_support(handle).is_good());
 
     REQUIRE(graph->build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
 
     // Use int8_t as proxy for fp8
     Surface<int8_t> X_gpu(n * c * d * h * w, false);
     Surface<int8_t> W_gpu(k * c * r * s * t, false);
     Surface<int8_t> Y_gpu(n * k * out_d * out_h * out_w, false);
 
     Surface<float> X_descale_gpu(1, false);
     Surface<float> W_descale_gpu(1, false);
     Surface<float> Y_scale_gpu(1, false);
     Surface<float> amax_gpu(1, false);
 
     int64_t workspace_size;
     REQUIRE(graph->get_workspace_size(workspace_size).is_good());
     Surface<int8_t> workspace(workspace_size, false);
 
     std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
         {X, X_gpu.devPtr},
         {W, W_gpu.devPtr},
         {Y, Y_gpu.devPtr},
         {descale_x, X_descale_gpu.devPtr},
         {descale_w, W_descale_gpu.devPtr},
         {scale_y, Y_scale_gpu.devPtr},
         {amax, amax_gpu.devPtr}};
 
     std::cout << graph->print() << std::endl;

     for (int i = 0; i < 10; ++i) {
        REQUIRE(graph->execute(handle, variant_pack, workspace.devPtr).is_good());
        // CUDNN_CHECK(cudnnDestroy(handle));
    }

 }