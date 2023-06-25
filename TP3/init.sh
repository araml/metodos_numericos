#!/usr/bin/env bash

git clone https://gitlab.com/libeigen/eigen
git clone https://github.com/pybind/pybind11
git clone https://github.com/doctest/doctest/

if [ -d "./csvs" ]
then
    echo "CSV folder already exists"
else
    mkdir csvs
fi

if [ -d "./figures" ]
then
    echo "Figure folder already exists"
else
    mkdir figures
fi