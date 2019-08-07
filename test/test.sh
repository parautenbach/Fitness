#!/bin/bash
echo "Checking style..."
./style.sh
echo "Checking doc strings..."
./docs.sh
echo "Linting..."
./lint.sh
echo "Checking code complexity..."
./cc.sh
echo "Checking code duplication..."
./cpd.sh
