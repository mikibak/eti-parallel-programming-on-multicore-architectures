#!/bin/bash

echo "==================== SYSTEM INFORMATION ===================="

echo -e "\n>>> Operating System <<<"
echo "Kernel: $(uname -r)"
echo "Architecture: $(uname -m)"
if command -v lsb_release &>/dev/null; then
    echo "Distro: $(lsb_release -d | cut -f2)"
    echo "Release: $(lsb_release -r | cut -f2)"
else
    grep PRETTY_NAME /etc/os-release
fi

echo -e "\n>>> CPU <<<"
cpu_model=$(lscpu | grep "Model name:" | cut -d: -f2 | sed 's/^[ \t]*//')
cpu_cores=$(lscpu | grep "^CPU(s):" | head -1 | awk '{print $2}')
cpu_threads=$(lscpu | grep "Thread(s) per core:" | awk '{print $4}')
echo "Model: $cpu_model"
echo "Cores: $cpu_cores"
echo "Threads per core: $cpu_threads"

echo -e "\n>>> GPU <<<"
if command -v nvidia-smi &>/dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)
    IFS=',' read -r gpu_name gpu_driver gpu_mem <<< "$gpu_info"
    echo "GPU: $gpu_name"
    echo "Driver: $gpu_driver"
    echo "Memory: $gpu_mem"
else
    echo "No NVIDIA GPU detected"
fi

echo -e "\n>>> CUDA <<<"
if command -v nvcc &>/dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    echo "CUDA Version: $cuda_version"
else
    echo "CUDA not installed"
fi

echo -e "\n>>> OpenMP <<<"
openmp_version=$(gcc -fopenmp -dM -E - < /dev/null | grep -i openmp | awk '{print $3}')
if [ -n "$openmp_version" ]; then
    echo "OpenMP Version: $openmp_version"
else
    echo "OpenMP not detected"
fi

echo -e "\n>>> MPI <<<"
if command -v mpirun &>/dev/null; then
    mpi_version=$(mpirun --version | head -n1)
    echo "$mpi_version"
else
    echo "MPI not installed"
fi

echo -e "\n==========================================================="
