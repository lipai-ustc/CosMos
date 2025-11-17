#!/bin/bash

# Install required Python dependencies
pip install numpy>=1.21.0
pip install ase>=3.22.1
pip install dscribe>=1.2.0

# Get repository directory
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PYTHONPATH_CONFIG="export PYTHONPATH=\"$REPO_DIR:\\$PYTHONPATH\""

# Add to .bashrc if not already present
if ! grep -qxF "$PYTHONPATH_CONFIG" ~/.bashrc; then
  echo "$PYTHONPATH_CONFIG" >> ~/.bashrc
  echo "Added repository path to ~/.bashrc"
else
  echo "Repository path already in ~/.bashrc"
fi

# Apply to current session
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
echo "Current session PYTHONPATH updated"

# Verify installation
echo "Dependency installation completed!"
python -c "import numpy, ase, dscribe; print('All dependencies imported successfully')"

# Check Python path configuration
echo "Current PYTHONPATH: $PYTHONPATH"

if [ $? -eq 0 ]; then
  echo "Dependency packages installed successfully!"
else
  echo "Dependency package installation failed! Please check if the dependency package versions are correct."
  exit 1
fi
