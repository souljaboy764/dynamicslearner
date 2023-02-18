#!/bin/bash
for filename in data/raw/bins/*.bin; do
    mc_bin_to_log $filename data/raw/csv/$(basename -- ${filename%%.*}).csv
done
