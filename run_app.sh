#!/bin/bash

# Setup script for Home Credit Default Risk Predictor app

echo "=========================================================="
echo "Home Credit Default Risk Predictor - Setup & Run"
echo "=========================================================="

# Create directories if they don't exist
mkdir -p data/lgbm
mkdir -p historical_data

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.x to continue."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error creating virtual environment. Make sure python3-venv is installed:"
        echo "sudo apt install python3-venv python3-full"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install streamlit pandas numpy matplotlib seaborn lightgbm scikit-learn

# Check for model files and create links if needed
echo "Checking for model files..."
if [ ! -d "data/lgbm" ] || [ -z "$(ls -A data/lgbm/*.pickle 2>/dev/null)" ]; then
    echo "No model files found in data/lgbm. Looking for models elsewhere..."
    
    # Check common locations
    model_found=0
    
    for dir in "data/lgbm" "lgbm" "models" "saved_models" "webapp/data/lgbm"; do
        if [ -d "$dir" ] && [ -n "$(ls -A $dir/*.pickle 2>/dev/null)" ]; then
            echo "Found models in $dir. Creating links..."
            for model in $dir/*.pickle; do
                ln -sf "$(realpath $model)" "data/lgbm/$(basename $model)"
                echo "  Linked $(basename $model)"
                model_found=1
            done
            break
        fi
    done
    
    # Search recursively as a last resort
    if [ $model_found -eq 0 ]; then
        echo "Searching for models recursively..."
        for model in $(find . -name "lgbm_model*.pickle" -o -name "*lgbm*.pickle" 2>/dev/null); do
            ln -sf "$(realpath $model)" "data/lgbm/$(basename $model)"
            echo "  Linked $(basename $model)"
            model_found=1
        done
    fi
    
    if [ $model_found -eq 0 ]; then
        echo "Warning: No model files found. The application may not work correctly."
    fi
fi

# Check for CSV data files
echo "Checking for data files..."
data_found=0

for file in $(find . -name "*application*train*.csv" 2>/dev/null | head -1); do
    if [ -n "$file" ]; then
        echo "Found application data: $file"
        data_found=1
        # Create link if it's not in the data directory
        if [[ "$file" != "./data/"* ]]; then
            mkdir -p data
            ln -sf "$(realpath $file)" "data/$(basename $file)"
            echo "  Linked to data/$(basename $file)"
        fi
        break
    fi
done

if [ $data_found -eq 0 ]; then
    echo "Warning: No application train data found. Some features may not work."
fi

# Start the application
echo "=========================================================="
echo "Starting Home Credit Default Risk Predictor..."
echo "=========================================================="
streamlit run fixed_home_credit_predictor.py