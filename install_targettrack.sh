#!/bin/bash

# Install PyTorch and related packages
conda install pytorch=1.11.0 torchvision torchaudio -c pytorch -y

# Install additional packages
conda install ipython=7.19.0 matplotlib=3.3.2 numpy=1.19.2 pandas=1.1.3 scikit-image=0.18.3 scikit-learn=0.23.2 scipy=1.7.3 tqdm=4.50.2 -y

# Install packages via pip
pip install umap-learn==0.5.3 sparse==0.13.0 nd2reader==3.2.3 PyQt5==5.12 pyqtgraph==0.11.1 opencv-python==4.5.1.48 opencv-python-headless==4.4.0.46 h5py==3.1.0 albumentations==0.5.2 connected-components-3d==2.0.0 torchvision==0.8.2 alphashape

# Navigate to the gmmreg-python source directory and install
cd gmmreg-python/src
pip install .

# Go back to the original directory
cd ../..

