#!/bin/bash

epsilon=0.0
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	for gamma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
	do
		python -m smartcab.agent $alpha $gamma $epsilon
	done
done

