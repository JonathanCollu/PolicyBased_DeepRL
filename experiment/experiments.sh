#!/bin/bash

# trial
echo "~~~trial~~~"
python experiments.py \
-evaluate \
-run_name trial \
-optimizer adam \
-optim_lr 1e-3 \
-optimizer_v adam \
-optim_lr_v 1e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 100 \
-n 10 \
-gamma 0.9 \
-baseline \
-entropy \
-entropy_factor 0.2 ;

    