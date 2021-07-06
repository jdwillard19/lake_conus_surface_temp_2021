   
lake_conus_surface_temp_2021

==============================

EA-LSTM neural network for lake surface temperature 

---------------

Project Organization 

------------

├── conda_env.yaml   
├── data   
│   ├── description.txt   
│   ├── processed   
│   ├── raw   
│   │   ├── data_release - data pulled from sciencebase here   
│   │   ├── feats - raw data of input drivers   
│   │   └── obs - raw data of observations   
│   └── static   
│       └── lists   
├── home   
├── hpc   
├── LICENSE   
├── metadata   
│   └── lake_metadata.csv   
├── models   
├── README.md   
├── requirements.txt   
├── results   
│   ├── ealstm_hyperparams.csv   
│   └── xgb_hyperparams.csv   
└── src   
    ├── data   
    │   ├── preprocess.   
    │   ├── pull_data.r   
    │   ├── pytorch_data_operations.py   
    │   └── write_NLDAS_xy_pairs.py   
    ├── evaluate   
    │   ├── EALSTM_error_estimation_and_output_single_fold.py   
    │   ├── linear_model_error_est.py   
    │   ├── predict_lakes_EALSTM_final.py   
    │   └── xgb_error_est.py   
    ├── hpc   
    │   ├── create_ealstm_err_est_jobs.py   
    │   ├── create_EALSTM_tune_jobs.py   
    │   ├── create_err_est_jobs.py   
    │   ├── create_final_output_jobs.py   
    │   ├── create_preprocess_jobs.py   
    │   ├── create_xgb_tune_jobs.py   
    ├── hyperparam   
    │   ├── EALSTM_hypertune.py   
    │   └── xgb_hypertune.py   
    ├── models   
    │   └── pytorch_model_operations.py   
    ├── oneoff   
    │   ├── compileErrEstResults.py   
    │   ├── final_output_rmse_check.py   
    └── train   
        └── EALSTM_final_model.py   

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
* (no HPC)   
    + `cd src/data/`  
    + `python write_NLDAS_xy_pairs.py 0 185550`  
    + `python preprocess.py 0 185550` (run ONLY after previous job finished)  
* (HPC*)  
    + `cd src/hpc/`    
    + `python create_preprocess_jobs.py`  (create jobs)  
    + `cd /hpc/`  
    + `source data_jobs1.sh` (submit jobs)  
    + `source data_jobs2.sh` (run ONLY after previous jobs have finished)  

4. (optional, defaults already enabled)  Do hyperparameter tuning for EA-LSTM and XGB (Gradient Boosting)  
`cd src/hyperparam`  
`python xgb_hypertune [fold #]` (enter numbers 1-5 to get it for each, enter values in ~/results/xgb_hyperparams.csv)  
`python EALSTM_hypertune [fold #]` (enter numbers 1-5 to get it for each, enter values in ~/results/ealstm_hyperparams.csv)  

5. Train linear model (LM), XGB, and EALSTM for each fold and estimate error through cross validation
`cd src/evaluate`  
`python EALSTM_error_estimation_and_output_single_fold.py [fold #]` (run for each fold 1-5)  
`python linear_model_error_est.py [fold #]`     
`python xgb_error_est.py [fold #]`  

6. Compile error estimations  
`cd src/oneoff/`
`python compileErrEst.py`

7. Train final EA-LSTM model   
`cd src/train/` 




