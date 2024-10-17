
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
import numpy
import os
import math

##########################  reference code 

def deepseek_fused_gating(
        logits: torch.Tensor,  # fp32, shape = [num_experts][s][b] , b and s both needed
        topk:  int = 6,
        normalize_expert_weight: bool = True,
        routed_scaling_factor: float = 1.0,  # when normalize_expert_weight false, use it to scale weight
        expert_capacity_number: int = 1, #  the return value of get_capacity() ; if is None, goto no capacity mode
    ):
    gates = F.softmax(logits, dim=1)
    tokens, num_experts = gates.size()

    #print(gates[:,108])

    expert_weights, indices = torch.topk(gates, topk, dim=1)
    if normalize_expert_weight:
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
    expert_weights = expert_weights*routed_scaling_factor
    # without capacity
    if expert_capacity_number is None:
        # shape: [num_token, topk]
        return expert_weights, indices

    # TopK selection, Maskout unused experts
    topk_masked_gates = torch.zeros_like(gates).scatter(1, indices, expert_weights)
    topk_mask = torch.zeros_like(gates).scatter(1, indices, 1)

    capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=expert_capacity_number, dim=0, sorted=True)

    #print(capacity_probs[:,117],capacity_indices[:,117])

    capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
   
    final_mask = torch.logical_and(topk_mask, capacity_mask)
    drop_mask = torch.logical_not(final_mask)
    exceed_mask = torch.gather(drop_mask, 1, indices)
    # shape: [num_token, topk]
    final_expert_weights = expert_weights * torch.logical_not(exceed_mask)
    final_indices = indices.clone().masked_fill_(exceed_mask, -1) #torch.iinfo(torch.long).max)

    num_local_tokens_per_expert = torch.histc(indices, bins=num_experts, min=0, max=num_experts)


    """Calculate the load balancing loss contribution."""
    scale = num_experts / (tokens * topk)
    l_aux = scale * torch.dot(num_local_tokens_per_expert.to(gates.dtype), gates.mean(dim=0))

    return final_expert_weights, final_indices, l_aux,capacity_probs, capacity_indices

#####################################  fused gating 

import my_cuda_extension

"""
说明：
1 目前只支持e=160,k=6
2 accu_level = 0 1 2 3，数越大精度越高，但是耗时增加。推荐 1 或 2。
3 indice的超过capacity的无效填充，torch里面是torch.iinfo(torch.long).max .
  cuda 因为是32bit的int输出，目前暂时填充的是-1 (0xffffffff)。可以根据需求修改

性能：
1 纯cuda kernel耗时，前向0.1ms，后向0.01ms。torch整体约0.4ms+
2 关于误差。排序先后的差别，在capacity结尾会有个别对不上的。可以打开test case的打印检查下，都是weight值几乎相当的
情况，cuda里多线程随机选中

安装包：
1 复制package到python安装目录的site-package
2 在site-package/easy-install.pth 里手工追加一下这个包的路径即可导入my_cuda_extension
3 以下的python类建议直接使用。若有修改需求可以先沟通。

todo：
在原代码的gating到all2all中间还有一些数据重排和copy等操作
下一个版本会继续融合一部分比较有收益的操作
"""


class MyCustomCUDAFunctionParam():
    def __init__(self, device, seq_len: int = 4096, n_routed_experts: int = 160,
        num_experts_per_tok: int = 6, routed_scaling_factor: float = 16.0,
        capacity: int = 1, accu_level: int = 1):

        self.seq_len = seq_len
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.capacity = capacity
        self.accu_level = accu_level
        self.device = device 
        self.temp_buffer_size = 0
        self.temp_buffer = None
        self.cap_factor = float(n_routed_experts)/float(seq_len*seq_len*num_experts_per_tok)


    def cal_temp_buffer_size(self):
        self.temp_buffer_size = self.seq_len * self.num_experts_per_tok * 4  + 1024*self.n_routed_experts
        return self.temp_buffer_size

    #### save tensor to avoid high frequency copy and memset
    def save_temp_buffer(self,temp_buffer):
        self.temp_buffer = temp_buffer

    def save_tensor0(self,tensor0):
        self.tensor0 = tensor0
    def save_tensor1(self,tensor1):
        self.tensor1 = tensor1
    def save_tensor2(self,tensor2):
        self.tensor2 = tensor2
    def save_tensor3(self,tensor3):
        self.tensor3 = tensor3
    def save_tensor4(self,tensor4):
        self.tensor4 = tensor4                



class MyCustomCUDAFunction(torch.autograd.Function ):
    @staticmethod
    def forward(ctx,inputs, param ):

        seq_len = param.seq_len
        num_experts_per_tok = param.num_experts_per_tok
        device = param.device
        n_routed_experts = param.n_routed_experts
        temp_buffer = param.temp_buffer

        weight_out    = param.tensor0
        topk_res  = param.tensor1
        loss = param.tensor2
        gates = param.tensor3
        histc = param.tensor4

        #rep = 1600
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)
        #start_event.record()
        #for i in range(rep):
        #my_cuda_extension.forward(
        my_cuda_extension.forward(
            temp_buffer.contiguous(),
            inputs.contiguous(), 
            topk_res.contiguous(),
            weight_out.contiguous(),
            loss.contiguous(),
            gates.contiguous(),
            histc.contiguous(),                 
            seq_len ,
            n_routed_experts ,
            num_experts_per_tok ,
            param.routed_scaling_factor ,
            param.capacity ,
            param.accu_level  )

        ctx.param = param
        ctx.save_for_backward(gates,histc,topk_res)

        #end_event.record()
        #torch.cuda.synchronize()
        #elapsed_time_ms = start_event.elapsed_time(end_event)/rep
        #print(f'Elapsed time2: {elapsed_time_ms} ms')

        return weight_out,loss,topk_res

    @staticmethod
    def backward(ctx, grad_weight_out,grad_loss,grad_ind):

        cap_factor = ctx.param.cap_factor
        routed_factor = ctx.param.routed_scaling_factor
        grad_input = torch.zeros(ctx.param.seq_len, ctx.param.n_routed_experts, dtype=torch.float32,device=ctx.param.device)
        
        gates,histc,topk_res = ctx.saved_tensors
        
        #grad_ind = None

        #rep = 1600
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event = torch.cuda.Event(enable_timing=True)
        #start_event.record()
        #for i in range(rep):
        #my_cuda_extension.backward(
        my_cuda_extension.backward(
            grad_loss.contiguous(), 
            grad_weight_out.contiguous(),
            histc.contiguous(),
            topk_res.contiguous(),
            gates.contiguous(),
            grad_input.contiguous(),
            routed_factor,
            cap_factor,
            ctx.param.seq_len,
            ctx.param.n_routed_experts,
            ctx.param.num_experts_per_tok  )


        #end_event.record()
        #torch.cuda.synchronize()
        #elapsed_time_ms = start_event.elapsed_time(end_event)/rep
        #print(f'Elapsed time3: {elapsed_time_ms} ms')

        return grad_input,None


class DroplessFusedGating(torch.nn.Module):
    def __init__(self,device, bsz: int = 1,seq_len: int = 4096, n_routed_experts: int = 160,
        num_experts_per_tok: int = 6, routed_scaling_factor: float = 16.0,
        capacity:int = 154, accu_level: int = 2) -> None:
        super().__init__()

        self.device = device
        self.param = MyCustomCUDAFunctionParam(device, bsz*seq_len, n_routed_experts,
            num_experts_per_tok, routed_scaling_factor,
            capacity, accu_level)

        self.temp_buffer_size = self.param.cal_temp_buffer_size()
        
        self.register_buffer("temp_buffer", torch.zeros(self.temp_buffer_size,
            dtype=torch.float32,device=self.device))

        self.param.save_temp_buffer(self.temp_buffer)

        self.topk_res    = torch.zeros(seq_len, num_experts_per_tok, dtype=torch.int32,device=device,requires_grad=False)
        self.weight_out  = torch.zeros(seq_len, num_experts_per_tok, dtype=torch.float32,device=device)
        self.loss = torch.zeros(1, dtype=torch.float32,device=device)
        self.gates = torch.zeros(seq_len, n_routed_experts, dtype=torch.float32,device=device,requires_grad=False)
        self.histc = torch.zeros( n_routed_experts, dtype=torch.int32,device=device,requires_grad=False)

        self.param.save_tensor0(self.weight_out)
        self.param.save_tensor1(self.topk_res)
        self.param.save_tensor2(self.loss)
        self.param.save_tensor3(self.gates)
        self.param.save_tensor4(self.histc)



    def forward(self, input):
        # todo , if param changed, and temp_buffer_size new > old, plz realloc 
        temp_buffer_size = self.param.cal_temp_buffer_size()
        if (temp_buffer_size>self.temp_buffer_size):
            del self.temp_buffer
            self.register_buffer("temp_buffer", torch.zeros(self.temp_buffer_size,
                dtype=torch.float32,device=self.device)) 
            self.temp_buffer_size = temp_buffer_size
            self.param.save_temp_buffer(self.temp_buffer)
        #todo, if reshaped, re alloc buffer for tensors

        return MyCustomCUDAFunction.apply(input,self.param)


### test code 
if __name__ == "__main__":
    ### test case
    tensor_type = torch.float32
    model_dim = 4096
    num_experts = 160
    topkk = 6
    cap_factor = 1.0


    input_folder = "../logits_dump/"
    output_flag = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries = os.listdir(input_folder)
    idx = 0
    cap_num = math.ceil((model_dim*topkk / num_experts) * cap_factor)
    #print(" !!!!cap ",cap_num)
    
    for file in entries:
        file_addr = input_folder + file
        #print(file_addr)

        folder = ""

        if output_flag:
            idx += 1
            folder = "./case_" + str(idx) +"/"
            
            if(not os.path.exists(folder) ):
                os.makedirs(folder)

        #input_folder = "../logits_dump/1727337411.250248.logits.pt"

        logits = torch.load(file_addr)
        logits = logits.to(device)
        logits = logits.requires_grad_()

        if(output_flag):
            logits = logits.to("cpu")
            logits.detach().numpy().tofile(folder+"logis.dat")

        aux_loss_alpha = 0.0001
        seq_aux = True

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        rep = 10000

        if (output_flag ):
            rep=1
        loss = 0
        for i in range(rep):
            weight_out,topk_res,loss,capacity_probs, capacity_indices = deepseek_fused_gating(
                    logits, 
                    topkk,
                    False,
                    16.0,  # when normalize_expert_weight false, use it to scale weight
                    cap_num #  the return value of get_capacity() ; if is None, goto no capacity mode
                )
            loss3x =  weight_out.mean() + loss  # + topk_res_new.to(float).mean()*0.00001
            loss3x.backward()
        #print(" !!!!!!!!!!loss is ",logits.grad)
    
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)/rep
        if(not output_flag):
            print(f'Elapsed time: {elapsed_time_ms} ms')

       
        newGate = DroplessFusedGating(device,
            capacity = cap_num,
            accu_level = 2)
        logits2 = torch.load(file_addr)
        logits2 = logits2.to(device)
        logits2 = logits2.requires_grad_()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
       
        for i in range(rep):
            weight_out_new,loss2,topk_res_new = newGate.forward(logits2)
            loss3 =  weight_out_new.mean() + loss2  # + topk_res_new.to(float).mean()*0.00001
            loss3.backward()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)/float(rep)
        if(not output_flag):
            print(f'Elapsed time2: {elapsed_time_ms} ms')

        #print(logits2.grad)
        #print(logits.grad)
        print(torch.nn.functional.cosine_similarity (logits.grad.unsqueeze(0), logits2.grad.unsqueeze(0) )  )

        
        #print not matched place
        #for i in range(topk_res.size(0)):
        #    for j in range(topk_res.size(1)):
        #        ref_idx = topk_res[i,j]
        #        find = 0
        #        #if i==80:
        #        #    print(" big data at ",weight_out[i,j], ref_idx)
        #        if ref_idx != -1:
        #            for k in range(topk_res_new.size(1)):
        #                #if i==80:
        #                #    print(" my big data at ",weight_out_new[i,k], topk_res_new[i,k])
        #                if ref_idx==topk_res_new[i,k]:
        #                    find = 1
        #                    break
        #            if (find==0):
        #                #capacity_probs[i,ref_idx]
        #                #capacity_indices[i,ref_idx]
        #                for cc in range(cap_num):
        #                    if (capacity_indices[cc,ref_idx]==i): 
        #                        print(" res lost token at ",i," topk at ",j," eid at ",ref_idx.to("cpu")," when data is ",weight_out[i,j].to("cpu"), " at ",cc)




