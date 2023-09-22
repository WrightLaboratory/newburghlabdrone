#!/bin/bash

for i in {0..141}
do
   python run_slices.py N $i
   python run_slices.py E $i
done 
