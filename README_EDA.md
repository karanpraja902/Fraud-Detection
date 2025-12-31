# EDA Analysis Execution Guide

## 🚀 Quick Start

Execute all EDA notebooks with a single command:

```bash
./run_eda_analysis.sh
```

## 📋 Prerequisites

Ensure you have:
- Python 3.8+ installed
- Git repository cloned with data files
- Basic Python packages (script will install missing ones)

## 🎯 Usage Examples

### Basic Execution
```bash
# Run all notebooks with default settings
./run_eda_analysis.sh
```

### Check Dependencies Only
```bash
# Verify all required packages are installed
./run_eda_analysis.sh --check-deps
```

### Verbose Mode
```bash
# See detailed execution output
./run_eda_analysis.sh --verbose
```

### Fast Execution
```bash
# Skip optional packages for faster execution
./run_eda_analysis.sh --fast
```

### Force Installation
```bash
# Auto-install missing dependencies
./run_eda_analysis.sh --force-install
```

## 📊 What the Script Does

### 1. **Dependency Management**
- ✅ Checks for required Python packages
- ✅ Installs missing dependencies automatically
- ✅ Warns about optional packages (XGBoost, LightGBM, Optuna)
- ✅ Validates package versions

### 2. **Data Validation**
- ✅ Verifies all required data files exist
- ✅ Checks file sizes and integrity
- ✅ Validates data directory structure

### 3. **Notebook Execution**
- ✅ Executes all 3 notebooks in sequence
- ✅ Handles both Jupyter notebook and Python script execution
- ✅ Provides real-time progress updates
- ✅ Implements timeout protection (30 minutes per notebook)

### 4. **Error Handling & Recovery**
- ✅ Automatic retry mechanisms
- ✅ Detailed error logging with troubleshooting info
- ✅ Fallback execution methods
- ✅ Comprehensive error reporting

### 5. **Performance Monitoring**
- ✅ Execution time tracking for each notebook
- ✅ Memory usage monitoring
- ✅ Performance bottleneck identification
- ✅ Resource optimization suggestions

## 📁 Output Structure

```
logs/
├── eda_execution_YYYYMMDD_HHMMSS.log      # Detailed execution log
├── execution_report_YYYYMMDD_HHMMSS.txt   # Summary report
└── ...

notebooks/
└── outputs/
    ├── 01_exploration_output.ipynb        # Executed exploration notebook
    ├── 02_baseline_model_output.ipynb     # Executed baseline models notebook
    └── 03_experiments_output.ipynb        # Executed experiments notebook
```

## 🔍 Understanding the Output

### Success Indicators
- ✅ **Green SUCCESS messages** - All notebooks executed successfully
- ✅ **Execution times** - Performance metrics for each notebook
- ✅ **Report generation** - Comprehensive summary with next steps

### Error Handling
- ⚠️ **WARNING messages** - Non-critical issues that don't stop execution
- ❌ **ERROR messages** - Critical issues that prevent completion
- 📝 **Detailed logs** - Specific error information for debugging

### Log Files
- **Execution Log**: Complete command output with timestamps
- **Error Details**: Specific error messages and stack traces
- **Performance Data**: Execution times and resource usage

## 🛠️ Troubleshooting

### Common Issues

#### **Missing Dependencies**
```bash
# Solution: Force installation
./run_eda_analysis.sh --force-install
```

#### **Data Files Missing**
```bash
# Check data directory structure
ls -la data/raw/
ls -la data/processed/

# Ensure creditcard.csv exists in data/raw/
```

#### **Memory Issues**
```bash
# Use fast mode to reduce memory usage
./run_eda_analysis.sh --fast --verbose
```

#### **Jupyter Issues**
```bash
# Install Jupyter if missing
pip install jupyter

# Or use script mode only
# (Script will automatically fallback to Python execution)
```

### Debug Mode
```bash
# Enable verbose output for detailed debugging
./run_eda_analysis.sh --verbose

# Check specific log files
tail -f logs/eda_execution_*.log
```

## ⏱️ Expected Execution Times

| Notebook | Expected Time | Description |
|----------|---------------|-------------|
| 01_exploration.ipynb | 3-5 minutes | Data loading and basic analysis |
| 02_baseline_model.ipynb | 8-12 minutes | Model training and evaluation |
| 03_experiments.ipynb | 10-20 minutes | Advanced techniques and optimization |

**Total Time**: 15-25 minutes on modern hardware

## 🎯 Next Steps After Completion

1. **Review the execution report** in `logs/execution_report_*.txt`
2. **Check output notebooks** in `notebooks/outputs/`
3. **Analyze performance metrics** and execution times
4. **Proceed to Step 2**: Integrate findings into automated pipelines

## 📞 Support

If you encounter issues:

1. **Check the log files** in `logs/` directory
2. **Run with verbose mode** for detailed output
3. **Verify data files** are properly located
4. **Ensure Python environment** is correctly set up

## 🚀 Ready to Execute?

```bash
# Make sure script is executable
chmod +x run_eda_analysis.sh

# Run the complete EDA analysis
./run_eda_analysis.sh

# Monitor progress in real-time
tail -f logs/eda_execution_*.log
```

The script will handle all complexity and provide detailed feedback throughout the execution process!
