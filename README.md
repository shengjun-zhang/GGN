<p align="center">
    <h1 align="center">Gaussian Graph Network</h1>
<p align="center">

<p align="center">
    <span class="author-block">
        <a href="https://shengjun-zhang.github.io/">Shengjun Zhang</a></span>,&nbsp;
    <span class="author-block">
        <a href="https://github.com/Barrybarry-Smith">Xin Fei</a></span>,&nbsp;
    <span class="author-block">
        <a href="https://liuff19.github.io/">Fangfu Liu</a></span>,&nbsp;
    <span class="author-block">
        <a href="https://github.com/SongHaixu/shx.github.io">Haixu Song</a></span>,&nbsp;
    <span class="author-block">
        <a href="https://duanyueqi.github.io/">Yueqi Duan</a></span>&nbsp;
</p>

<h3 align="center"><a href="https://arxiv.org/abs/2503.16338">Paper</a> | <a href="https://shengjun-zhang.github.io/GGN/">Project Page</a>
</h3>

<p align="center">
    <img src="figure/pipeline.png">
</p>

## Installation

To get started, create a conda virtual environment using Python 3.10+ and install the requirements:

```bash
conda create -n ggn python=3.10
conda activate ggn
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

### RealEstate10K and ACID

We follow the instructions of [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat) to preprocess datasets.

pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the RealEstate10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

If you would like to convert downloaded versions of the RealEstate10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools).

### DTU 

Download the preprocessed DTU data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).

Convert DTU to chunks by running `python src/scripts/convert_dtu.py --input_dir PATH_TO_DTU --output_dir datasets/dtu`


## Running the Code
### Evaluation on full test sets

We retrain and update the [pretrained models](https://drive.google.com/drive/folders/1UPZ16yOLVzqMWb62G_5LaCgzw1ZlWXTP), including `re10k_new.ckpt` and `acid_new.ckpt`. 
We also provide full test datasets of Re10k and ACID in the asserts, including `re10k_XXview_all.json` and `acid_XXview_all.json`.
Run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoint/re10k_new.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.index_path=assets/re10k_4view_all.json

# acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoint/acid_new.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.index_path=assets/acid_4view_all.json 
```

We provide the evaluation results as follows:

**ACID 4 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|20.13|0.733|0.252|262|
|MonoSplat|19.62|0.729|0.256|262|
|HiSplat|20.58|0.774|0.217|344|
|GGN|23.23|0.715|0.275|151|

**ACID 8 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|18.22|0.690|0.291|524|
|MonoSplat|16.89|0.645|0.325|524|
|HiSplat|16.35|0.729|0.250|688|
|GGN|22.79|0.696|0.286|214|

**ACID 16 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|17.64|0.672|0.313|1049|
|MonoSplat|15.79|0.582|0.385|1049|
|HiSplat|14.93|0.692|0.279|1376|
|GGN|23.00|0.704|0.285|303|

**Re10K 4 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|20.86|0.793|0.204|262|
|MonoSplat|20.21|0.783|0.210|262|
|HiSplat|22.68|0.847|0.153|344|
|GGN|21.77|0.766|0.229|144|

**Re10K 8 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|19.69|0.767|0.233|524|
|MonoSplat|18.22|0.719|0.270|524|
|HiSplat|20.93|0.833|0.168|688|
|GGN|21.66|0.757|0.240|210|

**Re10K 16 Views**
|Methods|PSNR|SSIM|LPIPS|Gaussians|
|-------|----|----|-----|---------|
|MVSplat|19.18|0.753|0.250|1049|
|MonoSplat|16.92|0.647|0.331|1049|
|HiSplat|20.42|0.794|0.208|1376|
|GGN|21.67|0.759|0.241|305



### Evaluation

Get the [pretrained models](https://drive.google.com/drive/folders/1UPZ16yOLVzqMWb62G_5LaCgzw1ZlWXTP), and save them to `/checkpoints`. Run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoint/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.index_path=assets/re10k_4view.json

# acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoint/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
dataset.view_sampler.index_path=assets/acid_4view.json 
```

To render videos from a pretrained model, run the following

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpont/re10k \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false
```

### Training

Get the [initialized models](https://drive.google.com/drive/folders/1UPZ16yOLVzqMWb62G_5LaCgzw1ZlWXTP), and save them to `/checkpoints`. Run the following:

```bash
# re10k
python -m src.main +experiment=re10k data_loader.train.batch_size=6 checkpointing.load=checkpoints/re10k_init.ckpt
# acid
python -m src.main +experiment=acid data_loader.train.batch_size=6 checkpointing.load=checkpoints/acid_init.ckpt
```

## BibTeX

```bibtex
@article{zhang2024GGN,
    title   = {Gaussian Graph Network: Learning Efficient and Generatlizable Gaussian Representations from Multi-view Images},
    author  = {Zhang, Shengjun and Fei, Xin and Liu, Fangfu and Song, Haixu and Duan, Yueqi},
    journal = {Advances in Neural Information Processing Systems (NeurIPS)},
    year    = {2024},
}
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and [MVSplat](https://github.com/donydchen/mvsplat). Many thanks to these two projects for their excellent contributions!
