# Concentration Soft Sensors Paper

This repository holds the conductivity models and case studies used to develop soft sensors for estimating real-time ion concentrations 
in aqueous solutions. The soft sensor is applied to two case studies: (1) conductivity predictions in electronic waste leachates
derived from cathode ray tube and organic light-emitting diode devices and (2) real-time estimation of ion concentrations in the 
permeate outlet of a membrane-based separation for the recovery of cobalt ions from an acidic solution representative of lithium-ion 
battery leachate.

The data used in both case studies are stored in the `Data` and `Raw_data` folders. The soft sensors conductivity prediction for the 
first case case study are in the `single_salt_conductivity.ipynb`, `binary_salt_conductivity.ipynb`, and `ternary_salt_experiments.ipynb` 
files, with figures stored in the `Single_salt`, `Binary_salts`, and `Ternary_salts` folders, respectively. Lastly, the ion concentration 
estimates of the soft sensors in the second case study are in the `soft_sensor_membrane_sep.ipynb` file, with the figures stored in the 
`Li_Co` folder. 

`conductivity.py` holds the Shedlovsky and mean spherical approximation models for conductivity predictions. The codes were run on 
Python 3.10.14. Conductivity sensor calibration is performed `probe_calibration.ipynb`.

The following section describes how to set up an environment capable of solving these problems.

## Making a Python environment

1.  Install NumPy and Pandas with your preferred package manager. You may install NumPy and Pandas with, for example, ``pip``:

```
pip install numpy pandas
```

2.  Install ``matplotlib``, ``scipy``, and ``mpl_toolkits``: 

```
pip install scipy matplotlib mpl_toolkits
```
