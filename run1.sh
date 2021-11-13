conda activate torch
python train.py --log_dir "decomposer_v1_1024_attn_NT" --batch_size 16 --normals True --epoch 50 --mask True --gpu "1"
python train.py --log_dir "decomposer_v1_1024_attn_NF" --batch_size 16 --epoch 50 --mask True --gpu "1"
