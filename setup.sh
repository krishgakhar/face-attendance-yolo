#!/bin/bash

echo "Creating environment..."
python3 -m venv env
source env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing basic requirements..."
pip install ultralytics opencv-python pandas numpy==1.24.4

echo "Installing PyTorch (Jetson compatible)..."
# Example for JetPack 4.x (adjust if needed)
pip install torch==1.10.0 torchvision==0.11.0

echo "Setup complete!"

#just run
#chmod +x setup.sh
#./setup.sh