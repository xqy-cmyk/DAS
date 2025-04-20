# DAS

# Overview

Our work is conducted on LandBench 1.0.

**The code for soil moisture prediction based DAS is hosted here(https://github.com/xqy-cmyk/DAS)**

### Installation

- Python3.9.13
- pytorch 1.13.1
- pandas 1.4.4
- numpy 1.22.0
- scikit-learn 1.0.2
- scipy 1.7.3
- matplotlib 3.5.2
- xarray 2023.1.0
- netCDF4 1.6.2
- learn2learn 0.1.7


The latest  can work in 

linux-Ubuntu 18.04.6

### Prepare Data

**The data is hosted here(https://doi.org/10.11888/Atmos.tpdc.300294) with the following directory structure**<br>

```
|----LandBench
|----|----1
|----|----|----atmosphere
|----|----|----|----1979
|----|----|----|----|----2m_temperature.nc
|----|----|----|----|----10m_u_component_of_wind.nc
|----|----|----|----|----10m_v_component_of_wind.nc
|----|----|----|----|----precipitation.nc
|----|----|----|----|----specific_humidity.nc
|----|----|----|----|----surface_pressure.nc
|----|----|----|----1980
...
...
|----|----|----|---- 2020
|----|----|----land_surface
|----|----|----|----1979
|----|----|----|----|----surface_theraml_radiation_downwards_w_m2.nc
|----|----|----|----|----surface_solar_radiation_downwards_w_m2.nc
|----|----|----|----|----soil_temperature_level_2.nc
|----|----|----|----1980
...
...
|----|----|----|---- 2020
|----|----|----constants
|----|----|----|----clay_0-5cm_mean.nc
|----|----|----|----sand_0-5cm_mean.nc
|----|----|----|----silt_0-5cm_mean.nc
|----|----|----|----landtype.nc
|----|----|----|----soil_water_capacity.nc
|----|----|----|----dem.nc
```

### Prepare Config File

The config file of our work is `DAS/DAS/config.py`

### Process data and train model

Run the following command in the directory of `DAS/DAS/` to process data and start training.

```
python main.py 
```

### Detailed analyzing

Run the following command in the directory of `DAS/DAS/` to get detailed analyzing.

```
python postprocess.py 
python post_test.py 
```

