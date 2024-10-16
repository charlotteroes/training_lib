#include <cassert>
#include <vector>
#include <iostream>

#include "cuda_runtime.h"
#include "torch/extension.h"

#include "ppl/common/types.h"
#include "ppl/kernel/llm/cuda/pmx/gelu.h"
#include "ppl/kernel/llm/cuda/pmx/vision_embedding.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_TENSOR(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

torch::Tensor
gelu(const torch::Tensor input, const torch::Tensor gate, bool approximate) {
    CHECK_CUDA_TENSOR(input);
    CHECK_CUDA_TENSOR(gate);
    assert(input.dtype() == torch::kFloat16);
    assert(gate.dtype()  == torch::kFloat16);

    int dims = input.sizes().size();
    std::vector<int64_t> dimensions;
    for (int i = 0; i < dims; i++) {
        dimensions.push_back(input.size(i));
    }
    ppl::common::TensorShape input_shape;
    input_shape.SetDataType(ppl::common::DATATYPE_FLOAT16);
    input_shape.Reshape(dimensions);

    torch::Tensor output = torch::zeros_like(input);
    ppl::kernel::llm::cuda::pmx::gelu(0, &input_shape,
        (void*)input.data_ptr<at::Half>(), (void*)gate.data_ptr<at::Half>(),
        approximate, (void*)output.data_ptr<at::Half>());

    return output;
}

torch::Tensor
vision_embedding(const torch::Tensor pixel_values, const torch::Tensor class_weight,
                 const torch::Tensor patch_weight, const torch::Tensor position_weight,
                 const torch::Tensor patch_bias, bool bias_term, int hidden_dim,
                 int patch_size) {
    CHECK_CUDA_TENSOR(pixel_values);
    CHECK_CUDA_TENSOR(class_weight);
    CHECK_CUDA_TENSOR(patch_weight);
    CHECK_CUDA_TENSOR(position_weight);
    if (bias_term) {
        CHECK_CUDA_TENSOR(patch_bias);
        assert(patch_bias.dtype()  == torch::kFloat16);
    }
    assert(pixel_values.dtype() == torch::kFloat16);
    assert(class_weight.dtype() == torch::kFloat16);
    assert(patch_weight.dtype() == torch::kFloat16);
    assert(position_weight.dtype() == torch::kFloat16);

    torch::Tensor null_tensor;
    ppl::kernel::llm::cuda::pmx::vision_embedding_config config;
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t cudnn_status = cudnnCreate(&cudnn_handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::cout << "cudnnCreate failed: " << cudnnGetErrorString(cudnn_status)
                  << std::endl;
        return null_tensor;
    }
    config.cudnn_handle = cudnn_handle;

    config.bias_term = bias_term;
    config.hidden_dim = hidden_dim;
    config.patch_size = patch_size;
    config.batch_size    = pixel_values.size(0);
    config.image_channel = pixel_values.size(1);
    config.image_size    = pixel_values.size(2);
    if (pixel_values.size(2) != pixel_values.size(3)) {
        std::cout << "an image with unequal height and width isn't supported."
                  << std::endl;
        return null_tensor;
    }

    auto pplcommon_status = ppl::kernel::llm::cuda::pmx::vision_embedding_preprocessing(config);
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        std::cout << "ppl::kernel::llm::cuda::pmx::vision_embedding_preprocessing() failed with error: "
                  << ppl::common::GetRetCodeStr(pplcommon_status);
        return null_tensor;
    }
    void* gpu_buffer;
    cudaMalloc((void**)&gpu_buffer, config.total_buffer_size);
    config.buffer_addr = gpu_buffer;

    int32_t num_positions = config.grid * config.grid + 1;
    torch::Tensor output_embeddings = torch::zeros({config.batch_size,
        num_positions, hidden_dim}, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0));
    pplcommon_status = ppl::kernel::llm::cuda::pmx::vision_embedding(
        0, config, pixel_values.data_ptr<at::Half>(),
        (void*)class_weight.data_ptr<at::Half>(),    // [hidden_dim]
        (void*)patch_weight.data_ptr<at::Half>(),    // [hidden_dim, image_channel, patch_size, patch_size]
        (void*)position_weight.data_ptr<at::Half>(), // [num_positions * hidden_dim]
        (void*)patch_bias.data_ptr<at::Half>(),      // [hidden_dim]
        (void*)output_embeddings.data_ptr<at::Half>()
    );
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        std::cout << "ppl::kernel::llm::cuda::pmx::vision_embedding() failed with error: "
                  << ppl::common::GetRetCodeStr(pplcommon_status);
        return null_tensor;
    }

    pplcommon_status =  ppl::kernel::llm::cuda::pmx::vision_embedding_postprocessing(config);
    if (pplcommon_status != ppl::common::RC_SUCCESS) {
        std::cout << "ppl::kernel::llm::cuda::pmx::vision_embedding_postprocessing() failed with error: "
                  << ppl::common::GetRetCodeStr(pplcommon_status);
        return null_tensor;
    }

    cudaFree(gpu_buffer);
    cudnn_status = cudnnDestroy(cudnn_handle);
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {
        std::cout << "cudnnDestroy failed: " << cudnnGetErrorString(cudnn_status)
                  << std::endl;
        return null_tensor;
    }

    return output_embeddings;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu", &gelu, "invoke ppl.kernel.llm.cuda/gelu");
    m.def("vision_embedding", &vision_embedding,
          "invoke ppl.kernel.llm.cuda/vision_embedding");
}
