#!/usr/bin/env bash

git clone https://gitlab.com/libeigen/eigen
git clone https://github.com/pybind/pybind11

# download images
if [ -d "./caras" ]
then
    echo "Data files already exist"
else
    wget https://www.dropbox.com/s/3glgtzgiyilo5nj/ImagenesCaras.zip?dl=1 -O ImagenesCaras.zip
    unzip ImagenesCaras.zip -d "./caras"
fi
