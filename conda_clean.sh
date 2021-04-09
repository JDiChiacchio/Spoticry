#!/bin/bash


#conda clean --yes --packages
n=1
while read line; do
    conda install $line
    n=$((n+1))
done < requirements.txt