
lake_conus_surface_temp_2021

==============================

EA-LSTM neural network for lake surface temperature 

---------------

Project Organization 

------------

TREE HERE

--------

Pipeline to run

-------------

1. Install necessary dependencies from yml file (Anaconda/miniconda must be installed for this), and activate conda environment. Works best on Linux (confirmed on CentOS 7(Core) and Manjaro 20.1).
`conda env create -f conda_env.yaml`  
`conda activate conda_env`

2. Pull data from USGS Sciencebase repository (may take a while, ~10GB download)
`cd src/data/`
`Rscript pull_data.r`

3. Run first preprocessing scripts with either HPC* or not (*high performance computing, recommended) (0 and 185550 are the starting and ending indices of the lakes to be processed as listed in ~/metadata/lake_metadata.csv)  
&nbsp;&nbsp;* (no HPC)   
    + `cd src/data/`  
    + `python write_NLDAS_xy_pairs.py 0 185550`  
    + `python preprocess.py 0 185550` (run ONLY after previous job finished)  
(HPC*)  
    + `cd src/hpc/`    
    + `python create_preprocess_jobs.py`  (create jobs)  
    + `cd /hpc/`  
    + `source data_jobs1.sh` (submit jobs)  
    + `source data_jobs2.sh` (run ONLY after previous jobs have finished)  

4.  
