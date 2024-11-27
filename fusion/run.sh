python train_iemocap.py configs/LMF.yml 
CUDA_VISIBLE_DEVICES=0 nohup python -u train_iemocap.py configs/LMF.yml >/dev/null 2>&1 &

python train_iemocap.py configs/TT_LMF.yml 
CUDA_VISIBLE_DEVICES=0 nohup python -u train_iemocap.py configs/LMF.yml >/dev/null 2>&1 &

python train_iemocap.py configs/TOMFUN.yml 
CUDA_VISIBLE_DEVICES=0 nohup python -u train_iemocap.py configs/TOMFUN.yml >/dev/null 2>&1 &


python train_iemocap_old.py configs/LMF.yml 