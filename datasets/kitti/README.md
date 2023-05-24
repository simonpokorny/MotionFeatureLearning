# KITTI raw dataset

### Download info 

For downloading all raw data from the KITTI websites, create a new folder, copy this script into the folder and execute it from the command line:

./raw_data_downloader.sh

It will download the zip files and extract them into a coherent data structure: Each folder contains all sequences recorded at a single day, including the calibration files for that day.

### Repo folder structure

    .
    ├── prepared             
    │   └── train          
    │       ├── 2011_10_03          
    │       ├──     ...      
    │       └── 2011_09_26            
    └── from_rigid_flow         # ???

