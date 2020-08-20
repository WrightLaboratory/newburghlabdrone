Directory contains the relevant scripts, modules, and notebooks to produce the BMX analysis plots I've made up to this point.

I put flights in dictionaries of the following structure:
dictionary_name['flight name'].fulldict['autos'][dish_num,time_sample,freq_sample]
at this point, you have a class that allows for: 
- easy plotting of 1D cuts in dB, lin. ie dictionary_name['flight name'].plot_y_lin(dish_number,freq_indx, lb = -0.5, ub = 0.5, params=True)
- plotting scatter plots. ie dictionary_name['flight name'].plotscatter(dish_number,freq_indx,)
- printing 2D gaussian fit parameters; recalling array of fit parameters. 
- accessing a full dictionary of flight/telescope data, where timestamps have been interpolated together. i.e. dictionary_name['flight name'].fulldict['autos'][dish_num,time_index,freq_index] for autocorrelation data


Generally these take the arguments(dish_number, freq_sample) (or (polarization, dish_num, freq_sample) if a Wednesday flight) to start, with additional optional arguments for 1D plotting. This is all illustrated in the Processing_Basics notebook. 

A few things to note:
I hard coded in polarization for Thursday flights (which were done one polarization at a time), so we'll only be processing co-pol data
Wed has both polarization options

What exactly is in this directory?

NOTEBOOKS:
1. Processing_Basics.ipynb
   - Intro to reading in data, plotting
   - Catalog of flights
   - Basic analysis of 2D fit parameters
2. Extra_Plots.ipynb
   - Onboard GPS comparison
   - RFI investigation
   - Diode removal
   - Common point tagging (just motivates that we should calibrate data)
3. BMX_Time_Offsets.ipynb
   - Investigation of timing offsets for a series of frequencies, dishes

SCRIPTS:
1. get_flights.py 
    - generated the flight dictionaries that are read into the notebooks (saved in .txt files)
2. getallfits.py
    - generated the 2D fits for each flight
3. getallfits-concatFiles.py
    - generated the 2D fits for combined data from same attenuation/polarization flight pairs

MODULES:
1. bmxdata.py
    -Deals with processing telescope data, was written by BMX people
2. BMX_Classes.py
    - Reads in data, puts it in class, includes some extra functions
3. TimeOffsetFns.py
    - Dump of functions used to plot in BMX_Time_Offsets notebook


Additionally, a number of txt files are referenced, which are on rubin (they store flight dictionaries for easy read in)
1. thurs_raw.txt
    - dictionary of thursday flights without timing correction applied
2. thurs_offsets.txt
    - dictionary of thursday flights with timing correction applied
3. wed_raw.txt
    - dictionary of Wed flights
4. gaussianfits.txt
    - dictionary of fit parameters for all dishes/frequencies from individual flights
5. concat_flights.txt
    -dictionary of flight pairs with common polarization/attenuation
6. gaussianfits_concat.txt
    - dictionary of fit parameters for all dishes/frequencies from paired flights
7. FLYXXX_offsets.txt
    - one for each flight, contain flight dictionaries for timing offsets from -1 to 1s in 0.1s intervals
8. fits_by_dish.txt
    - fit parameters for all flights/dishes at all frequencies

