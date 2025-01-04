cd fusion
conda activate zo_tonn

CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/dev/LMF.yml 
CUDA_VISIBLE_DEVICES=1 nohup python -u train_iemocap.py configs/dev/LMF.yml >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/dev/TT_LMF.yml 
CUDA_VISIBLE_DEVICES=1 nohup python -u train_iemocap.py configs/dev/TT_LMF.yml >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/dev/TOMFUN.yml 
CUDA_VISIBLE_DEVICES=1 nohup python -u train_iemocap.py configs/dev/TOMFUN.yml >/dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/dev/TOMFUN_test.yml --test_only

CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/inference/offline.yml --test_only
CUDA_VISIBLE_DEVICES=1 python train_iemocap.py configs/inference/noise_aware_offline.yml --test_only


# python train_iemocap_old.py configs/LMF.yml 
python train_iemocap_old.py configs/TT_LMF.yml