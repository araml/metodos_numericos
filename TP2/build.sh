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

# download images
if [ -d "./caras" ]
then
    echo "Data files already exist"
else
    wget https://www.dropbox.com/s/3glgtzgiyilo5nj/ImagenesCaras.zip?dl=1 -O ImagenesCaras.zip
    unzip ImagenesCaras.zip -d "./caras"
fi