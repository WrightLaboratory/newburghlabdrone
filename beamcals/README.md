# beamcals - BEAm Mapping CALibration System

Greetings and welcome to the software and analysis repository for Laura Newburgh's 21cm Drone Calibration group at Yale University. We are currently developing beam calibration techniques for 21cm instruments (like CHIME and HIRAX) using a drone-based transmitter. So far, we have mapped the beams of a single DSA-10 dish at the Owens Valley Radio Observatory (OVRO), the 4-dish BMX array at Brookhaven National Laboratory (BNL), a small testbed radio telescope at Yale's Leitner Family Observatory and Planetarium (LFOP), and an 8 element HIRAX analogue located at Green Bank Observatory (GBO).

Contained within this git repo are python packages/modules for data processing, analysis notebooks, and a logbook that I hope will serve as a project wiki.

Today (Nov. 11th, 2021) is the first large scale working release of the newly refactored code, and I'm excited to share the code I've written for this project. We plan to maintain this as an open source project. If you reuse code, please reference our repository on GitHub https://github.com/WrightLaboratory/newburghlabdrone/ and cite our work. To contact the authors, please will.tyndall at yale.edu

- William Tyndall [WT] - (Graduate Student)


## Cloning and Installation:

Prior to the initial release of this python package, we updated to the most recent available distributions of anaconda (v 4.10.1) and python (v 3.8.8). If possible, upgrading to this version is recommended for full functionality and minimal deprication errors. The only non-Anaconda package was pygeodesy at the time of release. All required packages are listed in the pip requirements file within the beamcals directory.

Begin by cloning this git repository (the stored sshkey method is recommended):
```
git clone git@github.com:WrightLaboratory/newburghlabdrone.git
```

Install beamcals (and its dependencies) using pip:
```
cd newburghlabdrone/beamcals/
pip install .
```

The beamcals package should now be installed and added to the path for importation from any directory. Now you're ready to view and run the `tutorials` notebooks to get started.


## Directory Structure:

1. `analysis` : ipython notebooks for data analysis
2. `bmx` : older work from BMX beam mapping flights
3. `dronehacks` : older work that contains outdated module scripts
4. `beamcals` : the current version of the beamcals python package
5. `liveplots` : ipython notebooks for quickly looking at data during acquisitions
6. `logbook` : the project logbook/wiki where writeups are stored
7. `tests` : ipython notebooks for developing module functions and debugging
7. `tutorials` : ipython notebooks that contain tutorials for new users (start here!)

## The 'beamcals' Python Package:

The core software package that supports our analysis is the 'beamcals' package. Here we will list and briefly describe the contained modules and functions contained within the package. At root level, the requirements, setup, LICENSE, and README.md files are found. 

### Modules:

Inside `beamcals/beamcals` you will find the following python modules.

To utilize these modules in an analysis notebook, something resembling the following import statement is recommended:
```
from beamcals import corr
from beamcals import concat
from beamcals import drone
from beamcals import bicolog
import beamcals.plotting_utils as pu
import beamcals.fitting_utils as fu
import beamcals.geometry_utils as gu
import beamcals.time_utils as tu
from beamcals.sites import site
gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')
``` 

#### `bicolog.py`
  * This module contains a model constructed from measurements of our bicolog transmitter.
  * class `Bicolog_Beam` : This class reads in data from range measurements and creates the beam model when initialized.
  * `Interpolate_D_ptf` : This is an N-d interpolator that interpolates the Directivity as a function of frequency and pointing.
  * `Plot_Beam_pcolormesh` : A plotting function for a map of the beam at a single frequency (or a range).
  * `Plot_Beam_Profiles` : A plotting function for the angular profile of the beam at a single frequency (or a range).

#### `concat.py`
  * This module contains the `CONCAT` class, which is used to concatenate the telescope and drone data. This is accomplished by interpolating the drone's position coordinates at each telescope timestamp. Beginning with the beamcals version 1.1, config files can now be used to store/load parameters found using iterative fit functions. Loading from these config files improves runtimes by bypassing iterative loops when the parameters can be found from previous results.
  * class `CONCAT` : initialized with only two inputs `CORRDATCLASS` and `DRONEDATCLASS`. Several variables and binary flags have now been added: `config_directory="/hirax/GBO_Analysis_Outputs/concat_config_files/"`,`output_directory='/hirax/GBO_Analysis_Outputs/'`,`load_yaml=True`,`save_traceback=True`,`traceback=True`). The default filepaths can be used when working on the rubin cluster, though for personal installations you must specify these directory paths. The binary flags for `traceback` and `save_traceback` are used when you desire print statements and verification plots to appear in the environment and save them to disk, respectively. The `load_yaml` binary flag is `True` when you wish to attempt to load parameters from previous analysis to bypass the iterative fit loops.
    * `Extract_Source_Pulses` : This function should only be used if the transmitter source was pulsed. This function then finds the best-fit source pulsing solution by correlating a square wave with the telescope visibility data. You must input the `Period` and `Dutycycle` of the pulse in microseconds. If the timing correction, period, and dutycycle can be read from the input config file, the iterative loop is skipped and the previous solution is applied.
    * `Perform_Background_Subtraction` : This function creates a 'background' (`V_bg`) and a 'background subtracted' (`V_bgsub`) visibility matrix from the pulsed source solution.
    * `Synchronization_Function` : This function synchronizes the input drone and input correlator files by applying a time offset to the drone timestamps and reinterpolating the spatial coordinates. By iterating through a coarse and a fine array of time offsets and maximizing the agreement with a 2d Gaussian, we find the appropriate shift in the time axes of both input files. Inputs to this function are `inputcorr` and `inputdrone`. The tunable variables and their default values for the fitting are: The min,max,step of the coarse time offset array: `coarse_params=[-10.0,10.0,0.2]`, The min,max,step of the fine time offset array: `fine_params=[-0.5,0.5,0.01]`, The channels and frequencies: `chans=np.arange(0,2)`,`freqs=np.arange(100,1024,150)`, The spatial coordinates around the main beam in meters: `FMB_coordbounds=[50.0,50.0,150.0]`, and the threshold percentile of the visibility matrix above which to exclude during the fit of the main beam to avoid saturated data points: `FMB_ampbound=0.999)`. See the tutorial for clarification and observe the verification plots using the `Synchronization_Verification_Plots` function from `plotting_utils.py`.
    * `Export_yaml` : Exports a .yaml config file for the specified concat class that will store all found parameters from the iterative fits. The file will be written to the `config_directory` specified when the concat class was initialized--this defaults to `/hirax/GBO_Analysis_Outputs/concat_config_files/` on Rubin. Indicator comments that are human readable will show which functions were run when the file was analyzed. Stored variables can be used in the future to reduce load times for data with previously found solutions. If a .yaml already exists for the drone and correlator file combination, a new "versioned" config file will be generated and stored. 
    * `Main_Beam_Fitting` : This function calls the `Fit_Main_Beam(inputconcat,chans,freqs,coordbounds=[50.0,50.0,150.0],ampbound=0.999)` function from `fitting_utils.py` and saves the best-fit 2DGauss and 2DAiry parameters for the non-saturated points within the main beam. This uses the generalized 2D Gaussian function, so beware of the axes and ellpiticity parameter degeneracy.

#### `corr.py`
  * class `Corr_Data` : This class is used to read in data from an iceboard correlator. When given a directory containing iceboard files, it will automatically isolate the autocorrelations indices, connect subsequent files together, apply a gain calibration, and generate time and frequency axes.

#### `drone.py`
  *  class `Drone_Data` : This class is used to read in data from the drone. This class currently supports older _processed.csv files and the raw datcon files. Support for the ublox gps files will be added in the near future. 

#### `fitting_utils.py`
  * This module contains functions frequently used for fitting data.
  * `Gauss` : A simple 1d Gaussian, with inputs `(x,a,x0,sigma,k)`.
  * `Airy_2d_LC_opt` : 2D Airy Disk function, minus input V matrix used in least squares routine.
  * `Airy_2d_LC_func` : Analytical form of 2D Airy Disk function , used for plotting.
  * `Gauss_2d_LC_opt` : 2D Gaussian function, minus input V matrix used in least squares routine.
  * `Gauss_2d_LC_func` : Analytical form of 2D Gaussian function, used for plotting.
  * `Fit_Main_Beam` : A channel by frequency fitting loop, applying cuts to find best-fit Airy Disk and Gaussian function in 2D for the unsaturated main beam.

#### `geometry_utils.py`
  * This module contains some frequently used functions for coordinate transforms.
  * `rot_mat` : this function generates a rotation matrix used for coordinate transformations.
  * `xyz_to_rpt` : This transforms local cartesian coordinates to polar coordinates.

#### `plotting_utils.py`
  * This module contains plotting functions for the `Corr_Data`, `Drone_Data` and `CONCAT` classes. There are a great many of them...
  * Corr_Data Plotting Functions:
    * `Plot_Waterfalls` : Waterfall plots of the correlator data, showing all channels, times, and freqs.
    * `Plot_Saturation_Maps` :  Waterfall plots where non-zero values imply digital saturation.
    * `Plot_Time_Series` : Time series of selected times and frequencies on all correlator channels.
    * `Plot_Spectra` : Spectra for each channel, plotted with a variable time-step gap.
    * `Plot_Gains_vs_Data` : The gain and initial data for each correlator channel.
  * Drone_Data Plotting Functions: 
    * `Plot_Drone_Coordinates` : The drones position coordinates in cartesian wrt the origin.
    * `Plot_Angular_Coordinates` : The drones position coordinates in polar wrt the origin.
    * `Plot_3d` : 3d Plots in local cartesian and geocentric cartesian coordinate systems
    * `Plot_Transmitter_Pointing` : A chart showing the drone transmitter vector at each position.
    * `Plot_Polar_Lines_of_Sight` : The receiver and transmitter respective coordinates for all times.
  * CONCAT_Data Plotting Functions:
    * `Plot_Beammap_LC` : This plotting function creates the beautiful beammaps!
    * `Synchronization_Verification_Plots`: This plot shows several indicators of Synchronization performance.

#### `time_utils.py`
  * This module contains a few utility functions used in the `drone.py` and `concat.py` modules:
  * `interp_time` : Annie's function for generating UTC datetimes from drone data.
  * `Pulsed_Data_Waveform` : The square wave function used to find when the transmitting source is on/off.
  
#### `sites/site.py`
  * The `site` class is contained within a `site` module within a `sites` directory, making the full module call `from beamcals.sites import site`. The `site` class is used to bundle the site data contained in `GBO_config.npz`  to initialize the geometric environments created by the other modules. 
  * To create a site-specific configuration file, utilize the notebook `sites/Write_Site_Config_NPZs.ipynb` and enter the site data according to the specified conventions. `GBO_config.npz` is one such example.
  * Below, the variable `gbosite` is defined to illustrate how to utilize the site class to import the data contained in a configuration.npz file:
```
from beamcals.sites import site
gbosite=site.site('../beamcals/beamcals/sites/GBO_config.npz')
```
