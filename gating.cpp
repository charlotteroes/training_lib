#include <torch/extension.h>
//#include <cuda_runtime.h>
#include "gating_lib.h"


extern "C"  int gating_py(
    torch::Tensor  temp_buffer,
    torch::Tensor  logits,
    torch::Tensor  topk_res,
    torch::Tensor  weight_out,
    torch::Tensor  loss,

    torch::Tensor  gates,
    torch::Tensor  histc,

    //params
    //int bsz ,
    int seq_len,
    int n_routed_experts,
    int num_experts_per_tok ,
    //bool norm_topk_prob ,
    float routed_scaling_factor ,
    //float aux_loss_alpha ,
    //bool seq_aux ,
    int capacity_num,
    //bool drop_and_pad ,
    int accu_level 
     )
{

    gating_param gating_prs;
    gating_prs.bsz = 1;
    gating_prs.seq_len = seq_len;
    gating_prs.n_routed_experts = n_routed_experts;
    gating_prs.num_experts_per_tok = num_experts_per_tok;
    gating_prs.norm_topk_prob = false;
    gating_prs.routed_scaling_factor = routed_scaling_factor;
    //gating_prs.aux_loss_alpha = aux_loss_alpha;
    //gating_prs.seq_aux = seq_aux;
    gating_prs.drop_and_pad = false;
    gating_prs.opt_level = accu_level;
    gating_prs.expert_capacity_number = capacity_num;

    //gates.data<scalar_t>()

    return gating_api(
        (void*)temp_buffer.data_ptr<float>(),
         (void*)&gating_prs,
        (float*)logits.data_ptr<float>(),
        (int*)topk_res.data_ptr<int>(),
        (float*)weight_out.data_ptr<float>(),
         (float*)loss.data_ptr<float>(),
        (float*)gates.data_ptr<float>(),
         (int*)histc.data_ptr<int>() 
           );
}


extern "C"  int gating_py_bk(
    torch::Tensor  grad_loss,
    torch::Tensor  grad_weight_out,
    torch::Tensor  histc,
    torch::Tensor  topk_res,
    torch::Tensor  gates,

    torch::Tensor  grad_input,
    //params

    float routed_factor ,
    float loss_factor ,
    int m,int e, int k
     )
{
    
    return gating_api_bk(
        (float*)grad_loss.data_ptr<float>(),
        (float*)grad_weight_out.data_ptr<float>(),
        (int*)histc.data_ptr<int>(),
         (int*)topk_res.data_ptr<int>(),
        (float*)gates.data_ptr<float>(),
         (float*)grad_input.data_ptr<float>() ,
         routed_factor,loss_factor, m,e,k
           );
}

PYBIND11_MODULE(my_cuda_extension, m) {
    m.def("forward", &gating_py, "Gating forward CUDA");
    m.def("backward", &gating_py_bk, "Gating backward CUDA");
}


