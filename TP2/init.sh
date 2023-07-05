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

if [ -d "./matrices" ]
then
    echo "Matrix folder already exists"
else
    mkdir matrices
fi

if [ -d "./figures" ]
then
    echo "Figure folder already exists"
else
    mkdir figures
fi


if [ -d "./csvs" ]
then
    echo "CSV folder already exists"
else
    mkdir csvs
fi