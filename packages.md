conda create -n 'warp' python=3.13
conda activate warp
conda install -c anaconda ipykernel -y
conda install nvidia/label/cuda-12.8.1::cuda-toolkit cudnn
pip install toml scipy numba tqdm h5py matplotlib ipywidgets ipympl imageio scikit-image imageio_ffmpeg seaborn pandas 
pip install warp-lang torch