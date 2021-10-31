#!/bin/sh
#Download the dataset

mkdir ckpt
mkdir runs
mkdir results


wget -q https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz

#Extract the dataset
tar xf ShelfImages.tar.gz

pip3 install requirements.txt