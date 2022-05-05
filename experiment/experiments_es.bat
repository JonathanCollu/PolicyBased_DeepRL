#!/bin/bash

python experiments.py \
-run_name ac_entropy_es_0 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-n 250 \
-use_es 0 ;

python experiments.py \
-run_name ac_entropy_es_1 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-n 250 \
-use_es 1