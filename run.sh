#!/bin/bash


gpu=$1
MODE=$2

if [ "$MODE" == "eval" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m test --cfg configs/mld_t2m.yaml --lambda_t2m 100 --lambda_m2m 100 --spm_path /data/wwj/ckpt/T5_SPM/SPM_H3D_Thr50.0_Temp1000_SAM1T0_E100.pth
elif [ "$MODE" == "mld" ]; then
    # TODO
    echo "Not implementation"
elif [ "$MODE" == "spm" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m ReAlignModule.train_spm --cfg configs/spm_t2m.yaml --NoiseThr 0.5 --maxT 1000 --step_aware M1T0 --CLThr 0.9 --CLTemp 0.1 | tee -a ./logs/h3d_tmr.log
elif [ "$MODE" == "tmr" ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -m eval_tmr --cfg configs/ft_mld_t2m.yaml | tee -a ./logs/tmr.log
else
    echo "Invalid MODE. Please choose from: test, train_mld, train_spm, eval_tmr"
fi
