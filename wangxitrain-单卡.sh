#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MASTER_PORT=$(python - <<'PY'
import socket,random,os
p=0
while p<20000 or p>40000:
    p=random.randint(20000,40000)
    try:
        socket.socket().bind(("127.0.0.1",p));print(p);break
    except OSError: pass
PY
)

python3 -m torch.distributed.run \
  --standalone \
  --nnodes 1 \
  --nproc_per_node 1 \
  --master_addr 127.0.0.1 \
  --master_port $MASTER_PORT \
  train.py \
    --deepspeed \
    --config /mnt/data/wangxi/diffusion-pipe-vace/examples/20250529医美vace训练/wan.toml
