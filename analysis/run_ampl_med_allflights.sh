#!/bin/bash

#for i in 648 649 535 
#do
#    python run_diffmaps_perflight.py  $i 620 18 35 512
#    python run_diffmaps_perflight.py  $i 620 18 35 640
#    python run_diffmaps_perflight.py  $i 620 18 35 768
#    python run_diffmaps_perflight.py  $i 620 18 35 896
#
#done

#for i in 619 625 646 647 536 533 
#do
#    python run_diffmaps_perflight.py  $i 618 20 40 512
#    python run_diffmaps_perflight.py  $i 618 20 40 640
#    python run_diffmaps_perflight.py  $i 618 20 40 768
#    python run_diffmaps_perflight.py  $i 618 20 40 896
#done

for i in {0..16}
do
    a=$((32 * $i + 512))
    for f in 648 649 535
    do 
       python run_diffmaps_perflight.py  $f 620 18 35 $a
    done 
    for f in 619 625 646 647 536 533
    do 
       python run_diffmaps_perflight.py  $f 618 20 40 $a
    done 
done 
