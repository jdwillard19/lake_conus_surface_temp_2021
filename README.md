
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

1. Install necessary dependencies from yml file (Anaconda must be installed for this), and activate conda environment. Works best on Linux (confirmed on CentOS 7(Core) and Manjaro 20.1).
`conda env create -f conda_env.yaml`  
`conda activate conda_env`

2. Pull data from USGS Sciencebase repository (may take a while, ~10GB download)
`cd src/data/`
`Rscript pull_data.r`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
