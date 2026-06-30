import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import simple_serve
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Change current working directory to project root so relative paths work
os.chdir(str(root_dir))

from simple_serve import app, load_model

# Load the model directly during function initialization
load_model()
