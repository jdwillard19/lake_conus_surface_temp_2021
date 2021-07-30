import os
import re
import pandas as pd
import pdb
import numpy as np
#######################################
# Nov 2020
# Jared - this script creates source model creation jobs to submit to msi in one script
# (note: takeme_source.sh must be custom made on users home directory on cluster for this to work: example script 
#        `
#          #!/bin/bash
#         source activate mtl_env
#         cd research/surface_temp/src/train
#         `  
#######################################

n_folds = 5


sbatch = ""
ct = 0
folds_arr = np.arange(5)+1

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp.out\n#SBATCH --error=EALSTM_%s_oversamp.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp.py %s"%(l)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp.sh'.format(l), 'w') as output:
        output.write(all)

compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)
