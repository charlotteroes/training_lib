#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include <float.h>


#define TOPK_DEFAULT 6
#define E_DEFAULT 160
#define MAX_PAD 0xffffffff


struct gating_param {
    int bsz ;
    int seq_len ;
    int n_routed_experts ;
    int num_experts_per_tok ;

    bool norm_topk_prob  ;
    float routed_scaling_factor ;
    float  aux_loss_alpha ;
    bool seq_aux ;

    // user def for capacity
	int expert_capacity_number ;
    int  drop_policy; //"probs", # or "position" or other
	bool drop_and_pad ;
    int opt_level;
};


extern "C"  int gating_api(
    void * temp_buf_in,
    void * params,
    float *logits,
    int * topk_res,
    float * weight_out,
    float * loss   ,
    float * gates_out,
    int * histc);

extern "C"  int gating_api_bk(
        float* grad_loss,
        float* grad_weight_out,
        int* histc,
        int* topk_res,
        float* gates,
        float* grad_input,
        float   routed_factor,
        float loss_factor,
        int m, int e, int k
    );

