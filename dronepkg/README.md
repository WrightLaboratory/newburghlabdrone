As of 8/20/20

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
    
    
![](https://github.com/dannyjacobs/ECHO/workflows/Run%20Tests/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/external-calibrator-for-hydrogen-arrays-echo/badge/?version=latest)](https://external-calibrator-for-hydrogen-arrays-echo.readthedocs.io/en/latest/?badge=latest)
[![dannyjacobs](https://circleci.com/gh/dannyjacobs/ECHO.svg?style=shield)](https://circleci.com/gh/dannyjacobs/ECHO)
[![codecov](https://codecov.io/gh/dannyjacobs/ECHO/branch/master/graph/badge.svg?token=X0UTMR10T2)](https://codecov.io/gh/dannyjacobs/ECHO)



The External Calibrator for Hydrogen Arrays (ECHO) is a system for calibrating wide-field radio frequency arrays using a radio transmitter mounted on a drone.
Primarily targeting (but not exclusively limited to) arrays operating in the sub-GHz band targeting highly redshifted
21cm radiation from the early universe.

This repository contains software for planning missions and analyzing data. It also contains hardware designs for drones and mounts.

This is an open source project. Use and reuse is encouraged.  If you use this code or hardware designs please reference the github repo.
http://github.com/dannyjacobs/ECHO and cite [our 2017 paper](http://adsabs.harvard.edu/abs/2017PASP..129c5002J).  If you have improvements please fork and send a PR!

## Community
* We have an active "Slack Connect" channel, accessible to anyone with a Slack account.  To be invited, email [Danny Jacobs](dcjacob2@asu.edu)
* Project [Web page](http://danielcjacobs.com/ECHO)
* [Code Documentation](https://external-calibrator-for-hydrogen-arrays-echo.readthedocs.io)
* A [mailing list](https://groups.google.com/d/forum/astro_echo) exists

## Installation
Install prerequisites. We recommend the anaconda package manager
* healpy (note that as of Jan 2020 healpy is not available for Windows)
* matplotlib
* numpy
* scipy

Get this repo
`git clone https://github.com/dannyjacobs/ECHO.git`

Install using pip
```
cd ECHO
pip install .
```


## Organization
The code is organized into a few modules. The beam mapping pipeline steps are
1. Read and down-select drone telemetry
2. Read and down-select radio telescope data (varies per telescope, usually spectra vs time)
3. Match up drone positions and telescope measurements. (Sometimes referred to as zippering.)
4. Grid beam (including power map, sample counts and standard deviation)
5. Analyze results. Example analysis steps include:
  1. subtract transmitter model
  2. plot beam maps
  3. plot slices
  4. plot drone events and dynamics
  5. difference beams
### Modules

 #### `plot_utils.py`
 Functions for plotting, but also all functions relating to healpix gridding
 and manipulation including gridding.
  * `grid_to_healpix` :  grids aligned RF power vs XYZ position
 into a healpix map
  * `make_beam` :  downselects desired spectral channel, converts from latlon to XYZ and calls `grid_to_healpix`
  * `project_healpix` :  flattens from spherical healpix map to 2d gnomic projection
  * `rotate_hpm` :  rotates a healpix map about the polar axis. useful for plotting
  * Other functions, most of whom are deprecated.
 #### `read_utils.py`
 Functions for reading and writing drone and beam data. Drone log file formats
 are not well documented and change all the time.
  * `read_map` : replacement for healpy read function that respects nans
  * `write_map`: replacement for healpy write function that respect nans
  * `apm_version`: tries to determine the version of ardupilot that wrote a log file.
  * `read_apm_log_A_B_C` : reads an ardupilot log of version A_B_C
    * returns position_times, positions,
    * attitude_times, attitudes,
    * waypoint_times, waypoint_numbers
  * `read_echo_spectrum` : reads a spectrum data file output by the ECHO spectrum logger ca 2017 (signalhound + get_sh_spectra)
  * `read_orbcomm_spectrum` : reads a spectrum data file output by the Bradley orbcomm system (ca 2017)
  * `channel_select` : given a spectrum, list of frequencies, and desired frequency returns closest frequency and spectrum amplitude
  * `interp_rx`: interpolates received power onto the measured position time grid
  * `flag_angles` : flags outlier yaws input times,yaw_angles return matching flag array
  * `flag_waypoints` : flags a range of time around a list of waypoints
  * `apply_flagtimes` : Given a list of bad times, a buffer size and a time array, generate a flag table.
#### `position_utils.py`
 * `latlon2xy` : about what you think
 * `to_spherical` : xyz to spherical
#### `time_utils.py`
Most of these are of dubious necessity.
 * `unix_to_gps` : thin wrapper around astropy.Time
 * `gps_to_HMS` : convert GPS time to Hours Minutes Seconds, thin wrapper around
#### `server_utils.py`
Stuff developed to support real-time operations. This never worked very well with mavlink.
### Scripts
#### Jacobs 2017
Scripts used in the 2017 paper are all run in [one master shell script](https://github.com/dannyjacobs/ECHO_paper1/blob/master/scripts/make_plots.sh)
 * plot_yaw.py
 * plot_GB_pos_power_interp.py
 * ECHO_zipper.py
 * ECHO_mk_beam.py
 * plot_ECHO_GB_power_rms_counts.py
 * plot_ECHO_GB_maps.py
 * plot_GB_slices.py
 * ECHO_sub_tx_beam.py
 * plot_ECHO_GB_ratios.py
 * plot_GB_avg_slices.py
 * plot_MWAtile_slices.py
 * MWATilemodel2hpm.py
 * combine_MWAtile_maps.py

 #### Utility Scripts
  * valon_readwrite.py : program the valon transmitter
  * gen_spherical_flight_path.py : generate waypoints in a healpix pattern
  * CST_to_healpix.py : convert the beam file from CST Microwave studio to a healpix map. Note that this should eventually be replaced with [pyuvbeam](https://github.com/RadioAstronomySoftwareGroup/pyuvdata/blob/master/pyuvdata/uvbeam.py).

