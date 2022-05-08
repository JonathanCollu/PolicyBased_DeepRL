#!/bin/bash

#reinforce
python experiments.py \
-run_name reinf \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 ;

#ac_bootstrap n 50
python experiments.py \
-run_name ac_n50 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-n 50 ;

#ac_bootstrap n 125
python experiments.py \
-run_name ac_n125 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-n 125 ;

#ac_bootstrap n 250
python experiments.py \
-run_name ac_n250 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg AC_bootstrap \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-n 250 ;

#ac_baseline
python experiments.py \
-run_name ac_bl \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-baseline ;

#ac_bootstrap_baseline
python experiments.py \
-run_name ac_bb \
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
-baseline ;

#reinforce entropy
python experiments.py \
-run_name reinf_entropy \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-entropy \
-entropy_factor 0.2 ;

#ac_baseline entropy
python experiments.py \
-run_name ac_bl_entropy \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-baseline \
-entropy \
-entropy_factor 0.2 ;

#ac_bootstrap entropy
python experiments.py \
-run_name ac_entropy \
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
-entropy \
-entropy_factor 0.2 ;

#ac_bootstrap_baseline entropy
python experiments.py \
-run_name ac_bb_entropy \
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
-baseline \
-entropy \
-entropy_factor 0.2 ;

#ac_baseline es0
python experiments.py \
-run_name reinf_es0 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-baseline \
-use_es 0 ;

#ac_baseline es1
python experiments.py \
-run_name reinf_es1 \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-baseline \
-use_es 1 ;

#ac_bootstrap es0
python experiments.py \
-run_name ac_es_0 \
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

#ac_bootstrap es1
python experiments.py \
-run_name ac_es_1 \
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
-use_es 1 ;

#ac_baseline es0 quantum
python experiments.py \
-run_name reinf_es0_quantum \
-optimizer adam \
-optim_lr 3e-3 \
-optimizer_v adam \
-optim_lr_v 3e-3 \
-device cpu \
-alg reinforce \
-traces 5 \
-trace_len 500 \
-epochs 500 \
-gamma 0.99 \
-baseline \
-use_es 0 \
-quantum ;
