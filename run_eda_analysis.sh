#!/bin/bash

# Fraud Detection EDA Analysis Runner
# Single command to execute all notebooks with comprehensive error handling and logging
# Usage: ./run_eda_analysis.sh [--verbose] [--check-deps] [--fast]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/eda_execution_$(date +%Y%m%d_%H%M%S).log"
DEPENDENCIES_FILE="${SCRIPT_DIR}/requirements.txt"
NOTEBOOKS_DIR="${SCRIPT_DIR}/notebooks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
VERBOSE=false
CHECK_ONLY=false
FAST_MODE=false
FORCE_INSTALL=false

# Performance tracking
TOTAL_START_TIME=$(date +%s)
NOTEBOOK_TIMES=()
ERRORS=()

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1" | tee -a "$LOG_FILE"
    fi
}

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Python packages with error handling
install_package() {
    local package="$1"
    local version="$2"
    
    print_verbose "Installing $package..."
    
    if [ -n "$version" ]; then
        pip install "$package==$version" 2>&1 | tee -a "$LOG_FILE"
    else
        pip install "$package" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Successfully installed $package"
        return 0
    else
        print_error "Failed to install $package"
        return 1
    fi
}

# Function to check and install dependencies
check_dependencies() {
    print_status "Checking Python dependencies..."
    
    # Create requirements list
    local required_packages=(
        "pandas"
        "numpy" 
        "scikit-learn"
        "matplotlib"
        "seaborn"
        "jupyter"
        "imbalanced-learn"
    )
    
    local optional_packages=(
        "xgboost"
        "lightgbm" 
        "optuna"
    )
    
    local missing_packages=()
    local optional_missing=()
    
    # Check required packages
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            missing_packages+=("$package")
        else
            print_verbose "✓ $package is available"
        fi
    done
    
    # Check optional packages
    for package in "${optional_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            optional_missing+=("$package")
        else
            print_verbose "✓ $package (optional) is available"
        fi
    done
    
    # Install missing required packages
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_warning "Missing required packages: ${missing_packages[*]}"
        
        if [ "$FORCE_INSTALL" = true ] || [ "$CHECK_ONLY" = false ]; then
            print_status "Installing missing packages..."
            
            for package in "${missing_packages[@]}"; do
                if ! install_package "$package"; then
                    print_error "Critical: Cannot proceed without $package"
                    return 1
                fi
            done
        else
            print_error "Missing required packages. Run with --force-install to auto-install."
            return 1
        fi
    else
        print_success "All required dependencies are available"
    fi
    
    # Install missing optional packages
    if [ ${#optional_missing[@]} -gt 0 ]; then
        print_warning "Missing optional packages: ${optional_missing[*]}"
        print_warning "Some advanced features may not be available"
        
        if [ "$FORCE_INSTALL" = true ]; then
            for package in "${optional_missing[@]}"; do
                install_package "$package" || print_warning "Could not install $package (optional)"
            done
        fi
    fi
    
    return 0
}

# Function to check data files
check_data_files() {
    print_status "Checking data files..."
    
    local data_files=(
        "${SCRIPT_DIR}/data/raw/creditcard.csv"
        "${SCRIPT_DIR}/data/processed/train.csv"
        "${SCRIPT_DIR}/data/processed/test.csv"
    )
    
    for data_file in "${data_files[@]}"; do
        if [ -f "$data_file" ]; then
            local file_size=$(stat -f%z "$data_file" 2>/dev/null || stat -c%s "$data_file" 2>/dev/null || echo "unknown")
            print_verbose "✓ Found $data_file (size: $file_size bytes)"
        else
            print_error "Missing data file: $data_file"
            print_error "Please ensure the data is properly prepared"
            return 1
        fi
    done
    
    print_success "All required data files are present"
    return 0
}

# Function to execute a notebook with error handling
execute_notebook() {
    local notebook_path="$1"
    local output_path="$2"
    local notebook_name=$(basename "$notebook_path")
    
    print_status "Executing $notebook_name..."
    local start_time=$(date +%s)
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$output_path")"
    
    # Execute notebook with timeout and error handling
    timeout 1800 jupyter nbconvert \
        --to notebook \
        --execute "$notebook_path" \
        --output "$output_path" \
        --ExecutePreprocessor.timeout=1200 \
        --ExecutePreprocessor.kernel_name=python3 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local execution_time=$((end_time - start_time))
    
    NOTEBOOK_TIMES+=("$notebook_name:$execution_time")
    
    if [ $exit_code -eq 0 ]; then
        print_success "$notebook_name completed in ${execution_time}s"
        
        # Check if output file was created
        if [ -f "$output_path" ]; then
            print_verbose "✓ Output saved to $output_path"
        else
            print_warning "Output file not found: $output_path"
        fi
        
        return 0
    else
        if [ $exit_code -eq 124 ]; then
            print_error "$notebook_name timed out after ${execution_time}s"
        else
            print_error "$notebook_name failed with exit code $exit_code"
        fi
        
        # Add to error list
        ERRORS+=("$notebook_name")
        
        # Try to extract error details
        if [ -f "$LOG_FILE" ]; then
            local error_details=$(tail -20 "$LOG_FILE" | grep -A 5 -B 5 "Error\|Exception\|Traceback" | tail -10)
            if [ -n "$error_details" ]; then
                print_error "Error details:"
                echo "$error_details" | tee -a "$LOG_FILE"
            fi
        fi
        
        return 1
    fi
}

# Function to convert notebooks to Python scripts and execute
execute_as_script() {
    local notebook_path="$1"
    local script_path="${notebook_path%.ipynb}.py"
    local notebook_name=$(basename "$notebook_path")
    
    print_status "Converting $notebook_name to Python script..."
    
    # Convert to Python script
    jupyter nbconvert --to python "$notebook_path" --output "$script_path" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        print_verbose "✓ Converted to $script_path"
        
        # Execute the script
        print_status "Executing $script_path..."
        python "$script_path" 2>&1 | tee -a "$LOG_FILE"
        
        local exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            print_success "$notebook_name executed successfully as script"
            return 0
        else
            print_error "$notebook_name script execution failed with exit code $exit_code"
            ERRORS+=("$notebook_name")
            return 1
        fi
    else
        print_error "Failed to convert $notebook_name to Python script"
        return 1
    fi
}

# Function to run notebooks
run_notebooks() {
    print_status "Starting notebook execution..."
    
    local notebooks=(
        "${NOTEBOOKS_DIR}/01_exploration.ipynb"
        "${NOTEBOOKS_DIR}/02_baseline_model.ipynb"
        "${NOTEBOOKS_DIR}/03_experiments.ipynb"
    )
    
    local outputs=(
        "${NOTEBOOKS_DIR}/outputs/01_exploration_output.ipynb"
        "${NOTEBOOKS_DIR}/outputs/02_baseline_model_output.ipynb"
        "${NOTEBOOKS_DIR}/outputs/03_experiments_output.ipynb"
    )
    
    # Create outputs directory
    mkdir -p "${NOTEBOOKS_DIR}/outputs"
    
    local success_count=0
    local total_notebooks=${#notebooks[@]}
    
    for i in "${!notebooks[@]}"; do
        local notebook="${notebooks[$i]}"
        local output="${outputs[$i]}"
        
        if [ ! -f "$notebook" ]; then
            print_error "Notebook not found: $notebook"
            ERRORS+=("$(basename "$notebook")")
            continue
        fi
        
        # Try notebook execution first
        if execute_notebook "$notebook" "$output"; then
            ((success_count++))
        else
            print_warning "Notebook execution failed, trying as Python script..."
            
            # Fallback to script execution
            if execute_as_script "$notebook"; then
                ((success_count++))
            else
                print_error "Both notebook and script execution failed for $(basename "$notebook")"
            fi
        fi
        
        # Add delay between notebooks to prevent resource conflicts
        sleep 2
    done
    
    print_status "Notebook execution completed: $success_count/$total_notebooks successful"
    return $success_count
}

# Function to generate execution report
generate_report() {
    local end_time=$(date +%s)
    local total_time=$((end_time - TOTAL_START_TIME))
    
    print_status "Generating execution report..."
    
    local report_file="${LOG_DIR}/execution_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "========================================"
        echo "FRAUD DETECTION EDA ANALYSIS REPORT"
        echo "========================================"
        echo ""
        echo "Execution Date: $(date)"
        echo "Total Execution Time: ${total_time}s"
        echo ""
        echo "Notebook Execution Times:"
        for time_info in "${NOTEBOOK_TIMES[@]}"; do
            echo "  $time_info"
        done
        echo ""
        echo "Errors Encountered:"
        if [ ${#ERRORS[@]} -eq 0 ]; then
            echo "  None - All notebooks executed successfully!"
        else
            for error in "${ERRORS[@]}"; do
                echo "  - $error"
            done
        fi
        echo ""
        echo "Log File: $LOG_FILE"
        echo "Output Directory: ${NOTEBOOKS_DIR}/outputs"
        echo ""
        echo "Next Steps:"
        echo "1. Review the execution logs in $LOG_FILE"
        echo "2. Check the output notebooks in ${NOTEBOOKS_DIR}/outputs"
        echo "3. Proceed to Step 2: Integrate EDA insights into automated pipelines"
        echo "========================================"
    } > "$report_file"
    
    print_success "Execution report saved to $report_file"
    cat "$report_file"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Execute all EDA notebooks with comprehensive error handling and logging."
    echo ""
    echo "OPTIONS:"
    echo "  --verbose           Enable verbose output"
    echo "  --check-deps        Check dependencies only, don't execute notebooks"
    echo "  --fast              Skip optional packages and use faster execution"
    echo "  --force-install     Force installation of missing dependencies"
    echo "  --help              Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                  Run all notebooks with default settings"
    echo "  $0 --verbose        Run with detailed output"
    echo "  $0 --check-deps     Check if all dependencies are available"
    echo "  $0 --fast           Quick execution with minimal dependencies"
}

# Function to cleanup on exit
cleanup() {
    print_verbose "Cleaning up temporary files..."
    # Add any cleanup logic here if needed
}

# Set up trap for cleanup
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --check-deps)
            CHECK_ONLY=true
            shift
            ;;
        --fast)
            FAST_MODE=true
            shift
            ;;
        --force-install)
            FORCE_INSTALL=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting Fraud Detection EDA Analysis"
    print_status "Log file: $LOG_FILE"
    log_with_timestamp "Starting EDA analysis execution"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Log script arguments
    log_with_timestamp "Script arguments: $*"
    
    # Step 1: Check dependencies
    if ! check_dependencies; then
        print_error "Dependency check failed"
        exit 1
    fi
    
    # Step 2: Check data files
    if ! check_data_files; then
        print_error "Data file check failed"
        exit 1
    fi
    
    # Step 3: Execute notebooks (unless check-only mode)
    if [ "$CHECK_ONLY" = false ]; then
        if ! run_notebooks; then
            print_error "Notebook execution completed with errors"
            print_warning "Check the log file for details: $LOG_FILE"
        else
            print_success "All notebooks executed successfully!"
        fi
    else
        print_success "Dependency check completed successfully"
        print_status "Use without --check-deps to execute notebooks"
        exit 0
    fi
    
    # Step 4: Generate report
    generate_report
    
    # Final status
    if [ ${#ERRORS[@]} -eq 0 ]; then
        print_success "EDA Analysis completed successfully!"
        print_status "All notebooks executed without errors"
        exit 0
    else
        print_error "EDA Analysis completed with ${#ERRORS[@]} error(s)"
        print_warning "Review the log file for debugging information"
        exit 1
    fi
}

# Run main function
main "$@"
