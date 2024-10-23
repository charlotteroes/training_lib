extern "C" int MLATrainRotaryPositionEmbeddingForwardAPI(
        const void* kv, 
        const void* k_pe,
        const void* cos, 
        const void* sin, 
        int64_t* position_ids, 
        int batch_size, 
        int seq_len,
        int num_heads, //num_attention_heads:128
        int v_head_dim,
        int qk_nope_head_dim, //128
        int qk_rope_head_dim, //64
        void *res_q,
        void *res_k);

extern "C" int MLATrainRotaryPositionEmbeddingBackwardAPI(
        // const void* src_grad_q, 
        const void* src_grad_kv,
        const void* cos, 
        const void* sin, 
        int64_t* position_ids, 
        int batch_size, 
        int seq_len,
        int num_heads, //num_attention_heads:128
        int v_head_dim,
        int qk_nope_head_dim, //128
        int qk_rope_head_dim, //64
        void *dst_grad_q,
        void *dst_grad_kv,
        void *dst_grad_k_pe);
