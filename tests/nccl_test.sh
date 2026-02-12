CUDA_VISIBLE_DEVICES=0,1 \
NCCL_IB_DISABLE=1 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
torchrun --standalone --nproc_per_node=2 \
  --rdzv-endpoint=127.0.0.1:29500 \
  --tee 3 --log_dir /tmp/torchrun_logs \
  - <<'PY'
import os, torch, torch.distributed as dist
print("rank env:", {k:os.environ.get(k) for k in ["RANK","LOCAL_RANK","WORLD_SIZE","MASTER_ADDR","MASTER_PORT"]}, flush=True)
rank=int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
print("before init_process_group", rank, flush=True)
dist.init_process_group("nccl")
print("after init_process_group", rank, flush=True)
PY
