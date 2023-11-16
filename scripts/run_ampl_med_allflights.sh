#!/bin/bash


#for i in {0..16}
#do
#    a=$((32 * $i + 512))
#    python run_diffmaps_perflight.py  623 618 20 40 $a
#done



for i in {0..16}
do
    a=$((32 * $i + 512))
    for f in 648 649 535
    do 
       python run_diffmaps_perflight.py  $f 620 18 35 $a
    done 
    for f in 619 625 623 646 647 536 533
    do 
       python run_diffmaps_perflight.py  $f 618 20 40 $a
    done 
done 

mv Flight_* /hirax/GBO_Analysis_Outputs/amplitude_corrections/ampsubfiles/
python combine_amplitudes.py 
