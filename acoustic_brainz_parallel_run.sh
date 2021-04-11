#!/bin/bash

# Call as nohup bash acoustic_brainz_parallel.sh to run without keeping ssh connection active.


nums="0 1 2 3 4 5 6 7 8 9 a b c d e f"
for val in $nums; do
    echo > ../acousticbrnz_out/"$val"
    python acoustic_brainz_db_constructor.py "$val" >> ../acousticbrnz_out/"$val" &
done
# Wait for all spawned processes to finish
wait
# Accumulate result stats in a summary file
echo > ../acousticbrnz_out/summary
for val in $nums; do
    cat ../acousticbrnz_out/"$val" |  tail -1 >> ../acousticbrnz_out/summary
done
# Combine database components
python acousticdb_merge.py &
wait
echo Done
