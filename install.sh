#!/bin/bash

# =============================================================================
# Automated ISIS & GDAL Installation Script
# This script handles the installation of the ISIS library
# =============================================================================

# Define your target environment name and data path here
ENV_NAME="isis_processing"
TARGET_ISISDATA="$HOME/isisdata"

# Prefer mamba if available, as conda can take over an hour to solve the ISIS package
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
    echo "Detected mamba. Installation will proceed faster."
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
    echo "Detected conda. Warning: Solving the ISIS environment may take a long time."
else
    echo "Error: Neither conda nor mamba is installed or in your PATH."
    exit 1
fi

# Ensure conda commands are natively available inside this bash script
eval "$($PKG_MGR shell.bash hook)"

echo "Step 1: Creating environment '$ENV_NAME' with ISIS and GDAL..."
$PKG_MGR create -y -n "$ENV_NAME" -c usgs-astrogeology -c conda-forge isis=9.0.0 gdal

# Ensure the data directory exists
if [ ! -d "$TARGET_ISISDATA" ]; then
    echo "Step 2: Creating ISISDATA directory at $TARGET_ISISDATA..."
    mkdir -p "$TARGET_ISISDATA"
else
    echo "Step 2: Found existing ISISDATA directory at $TARGET_ISISDATA."
fi

echo "Step 3: Configuring persistent path variables..."
# Get the absolute path of the newly created conda environment
ENV_PREFIX=$($PKG_MGR env list | grep "^$ENV_NAME " | awk '{print $NF}')

if [ -z "$ENV_PREFIX" ]; then
    echo "Error: Failed to locate the environment prefix. Cannot set variables."
    exit 1
fi

# The core fix for the ISISDATA reset issue.
# This binds the custom variables directly to the environment's activation hook.
conda env config vars set -n "$ENV_NAME" \
    ISISROOT="$ENV_PREFIX" \
    ISISDATA="$TARGET_ISISDATA"

echo ""
echo "========================================================"
echo "Installation and configuration successful!"
echo "========================================================"
echo "To use your environment and download the required planetary data, run:"
echo ""
echo "    conda activate $ENV_NAME"
echo "    downloadIsisData base $TARGET_ISISDATA"
echo "    downloadIsisData lro $TARGET_ISISDATA --exclude=\"kernels/**\""
echo "========================================================"
