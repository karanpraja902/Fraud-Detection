#!/bin/bash

# Simple EDA Analysis Runner - Fixed Version
# Single command to execute all notebooks with basic error handling

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
NOTEBOOKS_DIR="${SCRIPT_DIR}/notebooks"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create logs directory
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eda_execution_$(date +%Y%m%d_%H%M%S).log"

print_status "Starting EDA Analysis"
print_status "Log file: $LOG_FILE"

# Function to check dependencies
check_deps() {
    print_status "Checking dependencies..."
    
    local packages=("pandas" "numpy" "scikit-learn" "matplotlib" "seaborn" "jupyter" "imbalanced-learn")
    
    for package in "${packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_error "Missing package: $package"
            print_status "Installing $package..."
            pip3 install "$package" || {
                print_error "Failed to install $package"
                return 1
            }
        else
            print_status "✓ $package is available"
        fi
    done
    
    print_success "All dependencies are available"
}

# Function to run notebooks
run_notebooks() {
    print_status "Running notebooks..."
    
    local notebooks=(
        "${NOTEBOOKS_DIR}/01_exploration.ipynb"
        "${NOTEBOOKS_DIR}/02_baseline_model.ipynb"
        "${NOTEBOOKS_DIR}/03_experiments.ipynb"
    )
    
    # Create outputs directory
    mkdir -p "${NOTEBOOKS_DIR}/outputs"
    
    for notebook in "${notebooks[@]}"; do
        if [ ! -f "$notebook" ]; then
            print_error "Notebook not found: $notebook"
            continue
        fi
        
        local notebook_name=$(basename "$notebook")
        local output="${NOTEBOOKS_DIR}/outputs/$(basename "$notebook" .ipynb)_output.ipynb"
        
        print_status "Executing $notebook_name..."
        
        # Try Jupyter execution with explicit working directory
        if (cd "$SCRIPT_DIR" && jupyter nbconvert --to notebook --execute "$notebook" --output "$output" --ExecutePreprocessor.timeout=1200 2>&1) | tee -a "$LOG_FILE"; then
            print_success "$notebook_name completed"
        else
            print_error "$notebook_name failed, trying as Python script..."
            
            # Convert to Python and execute
            local script="${notebook%.ipynb}.py"
            jupyter nbconvert --to python "$notebook" --output "$script" 2>&1 | tee -a "$LOG_FILE"
            
            if python3 "$script" 2>&1 | tee -a "$LOG_FILE"; then
                print_success "$notebook_name executed as script"
            else
                print_error "$notebook_name script execution failed"
            fi
        fi
        
        sleep 2
    done
}

# Main execution
main() {
    echo "========================================" | tee -a "$LOG_FILE"
    echo "FRAUD DETECTION EDA ANALYSIS" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Started: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Change to project directory to ensure correct data paths
    print_status "Changing to project directory: $SCRIPT_DIR"
    cd "$SCRIPT_DIR"
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source train_env/bin/activate
    
    # Check dependencies
    check_deps
    
    # Run notebooks
    run_notebooks
    
    echo "" | tee -a "$LOG_FILE"
    echo "Completed: $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    print_success "EDA Analysis completed!"
    print_status "Check logs in: $LOG_FILE"
    print_status "Check outputs in: ${NOTEBOOKS_DIR}/outputs/"
}

# Run main
main "$@"
