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
# n_hid_arr = np.arange(25,275,25)
for i in range(300):
    #for each unique lake
    print(i)
    ct += 1
    # if not os.path.exists("../../../models/single_lake_models/"+name+"/PGRNN_basic_normAll_pball"): 
    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=split_%s.out\n#SBATCH --error=split_%s.err\n\n#SBATCH -p small"%(i,i)
    script = "source /home/kumarv/willa099/takeme_data.sh\n" #cd to directory with training script
    script2 = "python process_prism.py %s"%(i)
    # script3 = "python singleModel_customSparse.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_prism_%s.sh"%(i),sbatch])
    with open('../../hpc/job_prism_{}.sh'.format(i), 'w') as output:
        output.write(all)


compile_job_path= '../../hpc/prism_jobs.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)

print(ct, " jobs created, run this to submit: ", compile_job_path)