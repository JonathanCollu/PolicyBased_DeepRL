python experiments.py ^
-run_name ac_n50 ^
-optimizer adam ^
-optim_lr 4e-3 ^
-optimizer_v adam ^
-optim_lr_v 4e-3 ^
-device cpu ^
-alg AC_bootstrap ^
-traces 5 ^
-trace_len 500 ^
-epochs 300 ^
-n 50 ^
-gamma 0.99

python experiments.py ^
-run_name ac_n100 ^
-optimizer adam ^
-optim_lr 4e-3 ^
-optimizer_v adam ^
-optim_lr_v 4e-3 ^
-device cpu ^
-alg AC_bootstrap ^
-traces 5 ^
-trace_len 500 ^
-epochs 300 ^
-n 125 ^
-gamma 0.99

python experiments.py ^
-run_name ac_n200 ^
-optimizer adam ^
-optim_lr 4e-3 ^
-optimizer_v adam ^
-optim_lr_v 4e-3 ^
-device cpu ^
-alg AC_bootstrap ^
-traces 5 ^
-trace_len 500 ^
-epochs 300 ^
-n 250 ^
-gamma 0.99