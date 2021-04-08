#!/bin/bash

nums="0 1 2 3 4 5 6 7 8 9 a b c d e f"
for val in $nums; do
    nohup python acoustic_brainz_db_constructor.py "$val" > ../acousticbrnz_out 
done