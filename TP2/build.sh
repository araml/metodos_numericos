#!/bin/bash
mkdir -p build
cd build

# get number of available cores 
cores=$(nproc --all)

if ! command -v ninja &> /dev/null
then
    cmake -DUNITY=ON -DBUILD_TESTS=ON .. 
    make -j $cores
else
    cmake -DUNITY=ON -DBUILD_TESTS=ON -G Ninja .. 
    ninja
fi

