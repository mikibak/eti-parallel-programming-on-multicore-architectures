#!/bin/bash

echo "====================== SYSTEM INFORMATION ======================"
echo

echo ">>> Operating System <<<"
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release
echo

echo ">>> CPU <<<"
lscpu
echo

echo ">>> GPU <<<"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "No NVIDIA GPU detected"
fi
echo

echo ">>> CUDA <<<"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA not installed"
fi
echo

echo ">>> OpenMP <<<"
openmp_version=$(gcc -fopenmp -dM -E - < /dev/null | grep -i openmp | awk '{print $3}')
if [ -n "$openmp_version" ]; then
    echo "OpenMP version macro: $openmp_version"
else
    echo "OpenMP not detected"
fi
echo

echo ">>> MPI <<<"
if command -v mpirun &> /dev/null; then
    mpirun --version | head -n 1
else
    echo "MPI not installed"
fi

echo
echo "==============================================================="
