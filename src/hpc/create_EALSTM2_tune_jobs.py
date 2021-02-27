import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Jan 2021 - Jared - preprocess jobs
#         `  
#######################################



sbatch = ""
ct = 0
n_hid_arr = np.arange(25,400,25)
n_folds = 5
for i in n_hid_arr:
    for k in range(n_folds):
    #for each unique lake
        # print("i'i,k)

        # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
        header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --gres=gpu:k40:2\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=ealstm_tune_2layer_%shid_%sfold.out\n#SBATCH --error=ealstm_tune_2layer_%shid_%sfold.err\n\n#SBATCH -p k40"%(i,k,i,k)
        script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
        script2 = "python kfold_EALSTM_tune2.py %s %s"%(i,k)
        # script3 = "python singleModel_customSparse.py %s"%(l)
        all= "\n".join([header,script,script2])
        sbatch = "\n".join(["sbatch job_ealstm_tune_2layer_%shid_%sfold.sh"%(i,k),sbatch])
        with open('./jobs/job_ealstm_tune_2layer_%shid_%sfold.sh'%(i,k), 'w') as output:
            output.write(all)


compile_job_path= './jobs/ealstm2_tune_jobs.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)