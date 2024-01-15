#! /bin/bash

# full setup script
# change versions as needed

conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# # docker & multi GPU arch
# TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# # e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
# TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d

# create a folder in this directory and create a link to the preprocessed data folder
# cd Pointcept
# mkdir data
# ln -s ~/Desktop/datasets/preprocessed_norm_no_voxel_no_clutter/ ~/Desktop/Pointcept/data/cstdataset
