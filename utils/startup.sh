#!/bin/bash

# Update package manager
echo "Updating package manager..."
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

# Install Python 3.10 and dependencies
echo "Installing Python 3.10..."
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils python3-pip

# Install git if not already available
echo "Installing git..."
sudo apt-get install -y git

# Create Python 3.10 virtual environment
echo "Creating Python 3.10 virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

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

wget https://raw.githubusercontent.com/PrashCode5321/cable-routing-using-vla/refs/heads/main/api_server.py
python3 -m pip install fastapi uvicorn
python3 api_server.py  
