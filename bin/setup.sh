#!/bin/bash
if ! [ -x "$(command -v conda)" ]; then
  echo 'Error: it was not possible to find anaconda ( conda command ). Please install anaconda or miniconda for your system ( https://conda.io/miniconda.html ). Remember to source .bashrc to enable anaconda' >&2
  exit 1
fi
conda create --name pytorch python=3.6 --yes
source activate pytorch
conda install --yes pytorch torchvision -c pytorch
conda install --yes click
conda install --yes -c menpo opencv3
