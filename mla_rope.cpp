#include <torch/extension.h>
//#include <cuda_runtime.h>

#include "mla_rope_lib.h"


extern "C" int ppl_mla_rope_forward(
    torch::Tensor kv,
    torch::Tensor k_pe,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor position_ids,
    int batch_size, 
    int seq_len,
    int num_heads, //num_attention_heads:128
    int v_head_dim,
    int qk_nope_head_dim, //128
    int qk_rope_head_dim, //64
    torch::Tensor res_q,
    torch::Tensor res_k)
{
    return MLATrainRotaryPositionEmbeddingForwardAPI(
        (void*)kv.data_ptr<void>(),
        (void*)k_pe.data_ptr<void>(),
        (void*)cos.data_ptr<void>(),
        (void*)sin.data_ptr<void>(),
        (int64_t*)position_ids.data_ptr<int64_t>(),
        batch_size,
        seq_len,
        num_heads,
        v_head_dim,
        qk_nope_head_dim,
        qk_rope_head_dim,
        (void*)res_q.data_ptr<void>(),
        (void*)res_k.data_ptr<void>());
}

extern "C"  int ppl_mla_rope_backward(
    torch::Tensor src_grad_kv,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor position_ids,
    int batch_size, 
    int seq_len,
    int num_heads, //num_attention_heads:128
    int v_head_dim,
    int qk_nope_head_dim, //128
    int qk_rope_head_dim, //64
    torch::Tensor dst_grad_q,
    torch::Tensor dst_grad_kv,
    torch::Tensor dst_grad_k_pe)
{
    
    return MLATrainRotaryPositionEmbeddingBackwardAPI(
        (void*)src_grad_kv.data_ptr<void>(),
        (void*)cos.data_ptr<void>(),
        (void*)sin.data_ptr<void>(),
        (int64_t*)position_ids.data_ptr<int64_t>(),
        batch_size,
        seq_len,
        num_heads,
        v_head_dim,
        qk_nope_head_dim,
        qk_rope_head_dim,
        (void*)dst_grad_q.data_ptr<void>(),
        (void*)dst_grad_kv.data_ptr<void>(),
        (void*)dst_grad_k_pe.data_ptr<void>());
}

PYBIND11_MODULE(PPLMLARopeLib, m) {
    m.def("ppl_mla_rope_forward", &ppl_mla_rope_forward, "PPL MLA ROPE forward OPT kernel");
    m.def("ppl_mla_rope_backward", &ppl_mla_rope_backward, "PPL MLA ROPE backward OPT kernel");
}
