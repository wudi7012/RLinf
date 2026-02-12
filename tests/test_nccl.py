import os, torch, torch.distributed as dist

print("rank env:", {k:os.environ.get(k) for k in
      ["RANK","LOCAL_RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT"]}, flush=True)

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

print("before init_process_group", rank, "cuda", torch.cuda.current_device(), flush=True)
dist.init_process_group("nccl")
print("after init_process_group", rank, flush=True)

x = torch.ones(16, device="cuda") * (rank + 1)
dist.broadcast(x, 0)
dist.barrier()
if rank == 1:
    print("OK", x[0].item(), flush=True)

dist.destroy_process_group()

# 运行指令：
'''
 CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_SOCKET_IFNAME=eno2 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL torchrun --nproc_per_node=2   --master_addr=127.0.0.1 --master_port=29500   ./tests/test_nccl.py
 '''