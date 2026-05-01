#!/bin/bash
set -e

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

# Clone repositories
echo "Cloning cable-routing-using-vla repository..."
git clone https://github.com/PrashCode5321/cable-routing-using-vla.git

echo "Cloning openvla repository..."
git clone https://github.com/openvla/openvla.git

# Install requirements from first repo
echo "Installing requirements from cable-routing-using-vla..."
cd cable-routing-using-vla
pip install -r requirements.txt
cd ..

# Install second repo in editable mode
echo "Installing openvla in editable mode..."
cd openvla
pip install -e .
cd ..

echo "Setup complete!"

cp cable-routing-using-vla/openvla_utils/configs.py openvla/prismatic/vla/datasets/rlds/oxe/configs.py
cp cable-routing-using-vla/openvla_utils/transforms.py openvla/prismatic/vla/datasets/rlds/oxe/transforms.py
cp cable-routing-using-vla/openvla_utils/mixtures.py openvla/prismatic/vla/datasets/rlds/oxe/mixtures.py

# download model artifacts from Google Drive
python3 -m pip install gdown
gdown https://drive.google.com/uc?id=1__FlqKFAGThQWD2fYcB2Ts0R_4I2s7_8 -o artifacts.zip
unzip artifacts.zip
mv artifacts/content/openvla/ openvla/

# Find the run directory dynamically
RUN_DIR=$(ls -d openvla/runs/*/ | head -1 | sed 's:/$::')
echo "Found run directory: $RUN_DIR"
python3 cable-routing-using-vla/openvla_utils/patch_checkpoint_with_stats.py "$RUN_DIR" my_robot_dataset

cp cable-routing-using-vla/api.py openvla/api.py  
python3 -m pip install fastapi uvicorn
python3 api.py  