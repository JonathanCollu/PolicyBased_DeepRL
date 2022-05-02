#!/bin/bash

# trial
python experiments.py \
-run_name trial \
-optimizer adam \
-optim_lr 1e-3 \
-optimizer_v adam \
-optim_lr_v 1e-3 \
-device cuda \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 1000 \
-n 200 \
-gamma 0.9 \
-baseline \
-entropy \
-entropy_factor 0.2 \
-use_es 0

    