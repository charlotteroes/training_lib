import torch
import torch.nn.functional as F
from torch import nn
import time

import my_cuda_extension

class PPLMLATrainRotaryPositionEmbeddingLayer(nn.Module):
    def __init__(
        self,
        bsz: int, 
        seq_len: int, 
        num_heads: int = 128, 
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128, 
        qk_rope_head_dim: int = 64,
    ) -> None:
        super(PPLMLATrainRotaryPositionEmbeddingLayer, self).__init__()
        self.bsz = bsz
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim

    def forward(self, q, kv, k_pe, cos, sin, position_ids):
        res_kv = torch.empty((self.bsz, self.seq_len, 2, self.num_heads, (self.qk_nope_head_dim+self.qk_rope_head_dim)), 
            dtype=input.dtype, 
            device=input.device)
        my_cuda_extension.ppl_mla_rope_forward(
            kv.contiguous(), k_pe.contiguous(), cos.contiguous(), sin.contiguous(), position_ids.contiguous(), 
            self.bsz, self.seq_len, self.num_heads, self.v_head_dim, self.qk_nope_head_dim, self.qk_rope_head_dim,
            q.contiguous(), res_kv)
        return 

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)  # PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # CUDA 的随机种子（如果使用GPU）
    # random.seed(seed)  # Python 的随机种子
    # np.random.seed(seed)  # NumPy 的随机种子
    device = torch.device("cuda")
    host = torch.device("cpu")
    torch.set_default_dtype(torch.float16)

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

    # q.numpy().tofile('q.bin')
    # kv.numpy().tofile('kv.bin')
    # k_pe.numpy().tofile('k_pe.bin')
    cos.numpy().tofile('cos.bin')
    sin.numpy().tofile('sin.bin')
    position_ids.numpy().tofile('position_ids.bin')

    print("q ", q.dtype, q.shape)
    print("kv ", kv.dtype, kv.shape)
    print("k_pe ", k_pe.dtype, k_pe.shape)
    print("cos ", cos.dtype, cos.shape)
    print("sin ", sin.dtype, sin.shape)
    print("position_ids ", position_ids.dtype, position_ids.shape)

    q = q.to(device)
    kv = kv.to(device)
    k_pe = k_pe.to(device)
    cos = cos.to(device)
    sin = sin.to(device)
    position_ids = position_ids.to(device)

    model = PPLMLATrainRotaryPositionEmbeddingLayer().to(device)
    refq, refkv = model(
        q,
        kv,
        k_pe,
        cos,
        sin,
        position_ids,
        batch_size,
        seq_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim
    )

    print("refq ", refq.dtype, refq.shape)
    print("refkv ", refkv.dtype, refkv.shape)

    q.retain_grad()
    kv.retain_grad()
    k_pe.retain_grad()

    # loss = refq.sum() + refkv.sum()
    # loss.backward()

    external_grad_refq  = torch.rand_like(refq)
    external_grad_refkv = torch.rand_like(refkv)
    torch.autograd.backward([refq, refkv], [external_grad_refq, external_grad_refkv])

    print("external_grad_refq ", external_grad_refq.dtype, external_grad_refq.shape)
    print("external_grad_refkv ", external_grad_refkv.dtype, external_grad_refkv.shape)
    external_grad_refq.to(host).numpy().tofile('external_grad_q.bin')
    external_grad_refkv.to(host).numpy().tofile('external_grad_kv.bin')

    # print("q.grad:", q.grad)
    # print("kv.grad:", kv.grad)
    # print("k_pe.grad:", k_pe.grad)

    # refq = refq.to(host)
    # refkv = refkv.to(host)

    print("q.grad ", q.grad.dtype, q.grad.shape)
    print("kv.grad ", kv.grad.dtype, kv.grad.shape)
    print("k_pe.grad ", k_pe.grad.dtype, k_pe.grad.shape)
    q.grad.to(host).numpy().tofile('q_grad_ref.bin')
    kv.grad.to(host).numpy().tofile('kv_grad_ref.bin')
    k_pe.grad.to(host).numpy().tofile('k_pe_grad_ref.bin')
