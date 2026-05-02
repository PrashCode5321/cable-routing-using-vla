#!/bin/bash

# Update package manager
echo "Updating package manager..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/bin/activate
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n py310 python=3.10 -y
conda activate py310
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
/venv/py310/bin/python3 get-pip.py

# Clone repository
echo "Cloning openvla repository..."
git clone https://github.com/openvla/openvla.git
cd openvla

echo "Fetch changes from cable-routing-using-vla repo"
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/openvla_utils/configs.py
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/openvla_utils/mixtures.py
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/openvla_utils/transforms.py

mv configs.py prismatic/vla/datasets/rlds/oxe/
mv transforms.py prismatic/vla/datasets/rlds/oxe/
mv mixtures.py prismatic/vla/datasets/rlds/oxe/

# Install requirements from first repo
echo "Installing requirements from cable-routing-using-vla..."
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/requirements.txt
python3 -m pip install -r requirements.txt

# Install second repo in editable mode
echo "Installing openvla in editable mode..."
python3 -m pip install -e .

echo "Setup complete!"

# download model artifacts from Google Drive
echo "Fetch trained weights"
python3 -m pip install gdown
gdown https://drive.google.com/uc?id=1__FlqKFAGThQWD2fYcB2Ts0R_4I2s7_8
sudo apt install -y unzip
unzip folder_name.zip
mv content/openvla/runs/ .
rm -rf content/openvla/

# Find the run directory dynamically
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/openvla_utils/patch_checkpoint_with_stats.py
RUN_DIR=$(realpath $(ls -d runs/*/ | head -1 | sed 's:/$::'))
echo "Found run directory: $RUN_DIR"
python3 patch_checkpoint_with_stats.py "$RUN_DIR" my_robot_dataset

echo "Launch the FastAPI"
wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/api_server.py
python3 -m pip install fastapi uvicorn python-multipart pillow
python3 api_server.py --port 8000
