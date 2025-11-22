conda create --name kiss3dgen python=3.10
conda activate kiss3dgen
pip install -U pip

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post1

# Install Pytorch3D 
pip install iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# Install torch-scatter 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install other requirements
pip install -r requirements.txt

