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
  * This module contains the `CONCAT` class, which is used to concatenate the telescope and drone data. This is accomplished by interpolating the drone's position coordinates at each telescope timestamp.
  * class `CONCAT` : initialized with only two inputs `CORRDATCLASS` and `DRONEDATCLASS`.
    * `Extract_Source_Pulses` : This function should only be used if the transmitter source was pulsed. This function then finds the best-fit source pulsing solution by correlating a square wave with the telescope visibility data. You must input the `Period` and `Dutycycle` of the pulse in microseconds.
    * `Perform_Background_Subtraction` : This function creates a 'background' (`V_bg`) and a 'background subtracted' (`V_bgsub`) visibility matrix from the pulsed source solution.

#### `corr.py`
  * class `Corr_Data` : This class is used to read in data from an iceboard correlator. When given a directory containing iceboard files, it will automatically isolate the autocorrelations indices, connect subsequent files together, apply a gain calibration, and generate time and frequency axes.

#### `drone.py`
  *  class `Drone_Data` : This class is used to read in data from the drone. This class currently supports older _processed.csv files and the raw datcon files. Support for the ublox gps files will be added in the near future. 

#### `fitting_utils.py`
  * This module contains functions frequently used for fitting data.
  * `Gauss` : A simple 1d Gaussian, with inputs `(x,a,x0,sigma,k)`.

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