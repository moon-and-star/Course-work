#!/usr/bin/env bash
TRY_NUM=5
modes=(  "CoNorm" "AHE" "histeq" "imajust" "orig")
for j in "${modes[@]}"
do
	for k in $(seq 1 $TRY_NUM); do
		mv ./snapshots/experiment_22/RTSD/$j/trial_$k/snap_iter_2550.caffemodel \
			./snapshots/experiment_22/RTSD/$j/trial_$k/snap_iter_2070.caffemodel
		mv ./snapshots/experiment_22/RTSD/$j/trial_$k/snap_iter_2550.solverstate \
			./snapshots/experiment_22/RTSD/$j/trial_$k/snap_iter_2070.solverstate
	done
done