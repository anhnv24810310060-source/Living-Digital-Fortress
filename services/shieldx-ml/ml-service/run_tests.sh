#!/bin/bash
# Test runner for deep learning models

set -e

echo "==================================="
echo "Testing Deep Learning Models"
echo "==================================="

# Run unit tests
echo ""
echo "Running Autoencoder tests..."
python3 -m unittest tests.test_autoencoder -v

echo ""
echo "Running LSTM Autoencoder tests..."
python3 -m unittest tests.test_lstm_autoencoder -v

echo ""
echo "==================================="
echo "All tests passed!"
echo "==================================="
