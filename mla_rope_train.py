import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time

import my_cuda_extension

class PPLMLATrainRotaryPositionEmbeddingLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, k_pe, cos, sin, position_ids,
            bsz: int, 
            seq_len: int, 
            num_heads: int = 128, 
            v_head_dim: int = 128,
            qk_nope_head_dim: int = 128, 
            qk_rope_head_dim: int = 64,):
        res_kv = torch.empty((bsz, seq_len, 2, num_heads, (qk_nope_head_dim+qk_rope_head_dim)), 
            dtype=q.dtype, 
            device=q.device,
            requires_grad=kv.requires_grad)
        my_cuda_extension.ppl_mla_rope_forward(
            kv.contiguous(), k_pe.contiguous(), cos.contiguous(), sin.contiguous(), position_ids.contiguous(), 
            bsz, seq_len, num_heads, v_head_dim, qk_nope_head_dim, qk_rope_head_dim,
            q.contiguous(), res_kv)
        
        ctx.save_for_backward(cos, sin, position_ids)
        ctx.bsz = bsz
        ctx.seq_len = seq_len
        ctx.num_heads = num_heads
        ctx.v_head_dim = v_head_dim
        ctx.qk_nope_head_dim = qk_nope_head_dim
        ctx.qk_rope_head_dim = qk_rope_head_dim
        
        return q.contiguous(), res_kv

    @staticmethod
    def backward(ctx, grad_q, grad_kv):

        cos, sin, position_ids = ctx.saved_tensors
        bsz = ctx.bsz
        seq_len = ctx.seq_len
        num_heads = ctx.num_heads
        v_head_dim = ctx.v_head_dim
        qk_nope_head_dim = ctx.qk_nope_head_dim
        qk_rope_head_dim = ctx.qk_rope_head_dim
        
        dst_grad_kv = torch.empty((bsz, seq_len, num_heads, (qk_nope_head_dim+v_head_dim)), 
            dtype=q.dtype, 
            device=q.device)
        dst_grad_k_pe = torch.empty((bsz, seq_len, 1, qk_rope_head_dim), 
            dtype=q.dtype, 
            device=q.device)
        my_cuda_extension.ppl_mla_rope_backward(
            grad_kv.contiguous(), cos.contiguous(), sin.contiguous(), position_ids.contiguous(), 
            bsz, seq_len, num_heads, v_head_dim, qk_nope_head_dim, qk_rope_head_dim,
            grad_q.contiguous(), dst_grad_kv, dst_grad_k_pe)
        return grad_q, dst_grad_kv, dst_grad_k_pe, None, None, None, None, None, None, None, None, None

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)  # PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # CUDA 的随机种子（如果使用GPU）
    # random.seed(seed)  # Python 的随机种子
    # np.random.seed(seed)  # NumPy 的随机种子
    device = torch.device("cuda")
    host = torch.device("cpu")
    torch.set_default_dtype(torch.bfloat16)

    # 设置参数
    batch_size = 1
    num_heads = 128
    seq_len = 4096
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128

    q = torch.randn(batch_size, seq_len, num_heads, qk_nope_head_dim+qk_rope_head_dim, requires_grad=True)
    kv = torch.randn(batch_size, seq_len, num_heads, qk_nope_head_dim+v_head_dim, requires_grad=True)
    k_pe = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, requires_grad=True)

    # cos, sin = yarn_embed(v, seq_len=kv_seq_len) #TODO:
    # 随机生成角度（以弧度为单位）
    angles = torch.rand(seq_len, qk_rope_head_dim) * 2 * 3.14159
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    position_ids=torch.arange(seq_len).unsqueeze(0)
    # position_ids = kwargs.pop("indexes", 0) #TODO:

    external_grad_refq = torch.randn(batch_size, seq_len, num_heads, qk_nope_head_dim+qk_rope_head_dim)
    external_grad_refkv = torch.randn(batch_size, seq_len, 2, num_heads, qk_nope_head_dim+qk_rope_head_dim)

    # q.view(torch.uint16).numpy().tofile('q.bin')
    # kv.view(torch.uint16).numpy().tofile('kv.bin')
    # k_pe.view(torch.uint16).numpy().tofile('k_pe.bin')
    cos.view(torch.uint16).numpy().tofile('cos.bin')
    sin.view(torch.uint16).numpy().tofile('sin.bin')
    position_ids.view(torch.uint16).numpy().tofile('position_ids.bin')
    external_grad_refq.view(torch.uint16).numpy().tofile('external_grad_q.bin')
    external_grad_refkv.view(torch.uint16).numpy().tofile('external_grad_kv.bin')

    print("q ", q.dtype, q.shape)
    print("kv ", kv.dtype, kv.shape)
    print("k_pe ", k_pe.dtype, k_pe.shape)
    print("cos ", cos.dtype, cos.shape)
    print("sin ", sin.dtype, sin.shape)
    print("position_ids ", position_ids.dtype, position_ids.shape)

    q.detach().view(torch.uint16).numpy().tofile('q.bin')
    kv.detach().view(torch.uint16).numpy().tofile('kv.bin')
    k_pe.detach().view(torch.uint16).numpy().tofile('k_pe.bin')
    cos.detach().view(torch.uint16).numpy().tofile('cos.bin')
    sin.detach().view(torch.uint16).numpy().tofile('sin.bin')
    position_ids.detach().view(torch.uint16).numpy().tofile('position_ids.bin')

    q = q.to(device)
    kv = kv.to(device)
    k_pe = k_pe.to(device)
    cos = cos.to(device)
    sin = sin.to(device)
    position_ids = position_ids.to(device)
    external_grad_refq = external_grad_refq.to(device)
    external_grad_refkv = external_grad_refkv.to(device)

    model = PPLMLATrainRotaryPositionEmbeddingLayer()
    refq, refkv = model.apply(
        q,
        kv,
        k_pe,
        cos,
        sin,
        position_ids,
        batch_size, 
        seq_len, 
        num_heads, 
        v_head_dim, 
        qk_nope_head_dim, 
        qk_rope_head_dim
    )

    print("refq ", refq.dtype, refq.shape)
    print("refkv ", refkv.dtype, refkv.shape)
    refq.to(host).detach().view(torch.uint16).numpy().tofile('refq.bin')
    refkv.to(host).detach().view(torch.uint16).numpy().tofile('refkv.bin')

    q.retain_grad()
    kv.retain_grad()
    k_pe.retain_grad()

    # loss = refq.sum() + refkv.sum()
    # loss.backward()

    # external_grad_refq = torch.from_numpy(np.fromfile("../backward/external_grad_q.bin", dtype=np.float16)).reshape(refq.shape)
    # external_grad_refkv = torch.from_numpy(np.fromfile("../backward/external_grad_kv.bin", dtype=np.float16)).reshape(refkv.shape)
    torch.autograd.backward([refq, refkv], [external_grad_refq, external_grad_refkv])

    print("external_grad_refq ", external_grad_refq.dtype, external_grad_refq.shape)
    print("external_grad_refkv ", external_grad_refkv.dtype, external_grad_refkv.shape)

    # print("q.grad:", q.grad)
    # print("kv.grad:", kv.grad)
    # print("k_pe.grad:", k_pe.grad)

    print("q.grad ", q.grad.dtype, q.grad.shape)
    print("kv.grad ", kv.grad.dtype, kv.grad.shape)
    print("k_pe.grad ", k_pe.grad.dtype, k_pe.grad.shape)
    q.grad.to(host).view(torch.uint16).numpy().tofile('q_grad_ref.bin')
    kv.grad.to(host).view(torch.uint16).numpy().tofile('kv_grad_ref.bin')
    k_pe.grad.to(host).view(torch.uint16).numpy().tofile('k_pe_grad_ref.bin')
