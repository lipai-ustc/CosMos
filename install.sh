#!/bin/bash

# Install required Python dependencies
pip install numpy>=1.21.0
pip install ase>=3.22.1
pip install dscribe>=1.2.0

# Add repository path to PYTHONPATH
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "Added repository path $REPO_DIR to PYTHONPATH"

# Verify installation
echo "Dependency installation completed!"
python -c "import numpy, ase, dscribe; print('All dependencies imported successfully')"

if [ $? -eq 0 ]; then
  echo "Dependency packages installed successfully!"
else
  echo "Dependency package installation failed! Please check if the dependency package versions are correct."
  exit 1
fi