# CUDA_VISIBLE_DEVICES=3 python -m src.main +experiment=re10k \
# checkpointing.load=outputs/2024-05-11/00-07-44/checkpoints/epoch_9-step_75000.ckpt \
# mode=test \
# dataset/view_sampler=evaluation \
# test.compute_scores=true \
# dataset.view_sampler.index_path=assets/re10k_4view.json


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=acid \
checkpointing.load=/data1/zsj/PixelGS/new/effsplat/outputs/2024-05-11/00-00-38/checkpoints/epoch_54-step_75000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.index_path=assets/acid_4view.json