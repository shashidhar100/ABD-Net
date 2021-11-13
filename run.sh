conda activate torch
python train.py --log_dir "decomposer_v1_1024_max_NT" --batch_size 16 --normals True --epoch 50
# python train.py --log_dir "decomposer_v1_1024_attn_NF" --batch_size 16 --epoch 50
