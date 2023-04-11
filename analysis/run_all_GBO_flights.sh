#!/bin/bash

for i in {0..12} 
do
    python run_one_flight.py $i
done

cd /hirax/GBO_Analysis_Outputs
mv /hirax/GBO_Analysis_Outputs/FLY* /hirax/GBO_Analysis_Outputs/main_beam_fits

