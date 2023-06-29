#!/usr/bin/env bash
mkdir -p build
cd build

# get number of available cores 
cores=4

if [[ "$OSTYPE" == "darwin"* ]]; then
    cores=$(sysctl -n hw.physicalcpu)
else
    cores=$(nproc --all)
fi

if ! command -v ninja &> /dev/null
then
    cmake -DUNITY=ON -DBUILD_TESTS=ON .. 
    make -j $cores
else
    cmake -DUNITY=ON -DBUILD_TESTS=ON -G Ninja .. 
    ninja
fi

echo "Copying iterative_methods to Python folder"
cp iterative_methods.cpython* ../py/
echo "Copied iterative_methods to Python folder"
