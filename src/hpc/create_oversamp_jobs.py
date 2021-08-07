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

# for name in folds_arr:
#     ct += 1
#     #for each unique lake
#     print(name)
#     l = name

#     header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp1.out\n#SBATCH --error=EALSTM_%s_oversamp1.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
#     script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
#     script2 = "python EALSTM_err_est_oversamp1.py %s"%(l)
#     # script3 = "python singleModel_customSparse.py %s"%(l)
#     all= "\n".join([header,script,script2])
#     sbatch = "\n".join(["sbatch job_%s_foldEAoversamp1.sh"%(l),sbatch])
#     with open('../../hpc/job_{}_foldEAoversamp1.sh'.format(l), 'w') as output:
#         output.write(all)

# compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp1.sh'
# with open(compile_job_path, 'w') as output2:
#     output2.write(sbatch)
# print(compile_job_path)


for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n\
                              #SBATCH --mem=20g\n\
                              #SBATCH --mail-type=ALL\n\
                              #SBATCH --mail-user=willa099@umn.edu\n\
                              #SBATCH --output=EALSTM_%s_oversamp2.out\n\
                              #SBATCH --error=EALSTM_%s_oversamp2.err\n\
                              #SBATCH --gres=gpu:k40:1\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp2.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp2.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp2.sh'.format(l), 'w') as output:
        output.write(all)

compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp2.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)

sbatch = ""
ct = 0

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp3.out\n#SBATCH --error=EALSTM_%s_oversamp3.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp3.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp3.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp3.sh'.format(l), 'w') as output:
        output.write(all)



compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp3.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)

# sbatch = ""
# ct = 0

# for name in folds_arr:
#     ct += 1
#     #for each unique lake
#     print(name)
#     l = name

#     header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp4.out\n#SBATCH --error=EALSTM_%s_oversamp4.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
#     script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
#     script2 = "python EALSTM_err_est_oversamp4.py %s"%(l)
#     all= "\n".join([header,script,script2])
#     sbatch = "\n".join(["sbatch job_%s_foldEAoversamp4.sh"%(l),sbatch])
#     with open('../../hpc/job_{}_foldEAoversamp4.sh'.format(l), 'w') as output:
#         output.write(all)



# compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp4.sh'
# with open(compile_job_path, 'w') as output2:
#     output2.write(sbatch)
# print(compile_job_path)

# sbatch = ""
# ct = 0

# for name in folds_arr:
#     ct += 1
#     #for each unique lake
#     print(name)
#     l = name

#     header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp1b.out\n#SBATCH --error=EALSTM_%s_oversamp1b.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
#     script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
#     script2 = "python EALSTM_err_est_oversamp1b.py %s"%(l)
#     all= "\n".join([header,script,script2])
#     sbatch = "\n".join(["sbatch job_%s_foldEAoversamp1b.sh"%(l),sbatch])
#     with open('../../hpc/job_{}_foldEAoversamp1b.sh'.format(l), 'w') as output:
#         output.write(all)



# compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp1b.sh'
# with open(compile_job_path, 'w') as output2:
#     output2.write(sbatch)
# print(compile_job_path)

sbatch = ""
ct = 0

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp1c.out\n#SBATCH --error=EALSTM_%s_oversamp1c.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp1c.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp1c.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp1c.sh'.format(l), 'w') as output:
        output.write(all)



compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp1b.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)

# sbatch = ""
# ct = 0

# for name in folds_arr:
#     ct += 1
#     #for each unique lake
#     print(name)
#     l = name

#     header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp5.out\n#SBATCH --error=EALSTM_%s_oversamp5.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
#     script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
#     script2 = "python EALSTM_err_est_oversamp5.py %s"%(l)
#     all= "\n".join([header,script,script2])
#     sbatch = "\n".join(["sbatch job_%s_foldEAoversamp5.sh"%(l),sbatch])
#     with open('../../hpc/job_{}_foldEAoversamp5.sh'.format(l), 'w') as output:
#         output.write(all)



# compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp5.sh'
# with open(compile_job_path, 'w') as output2:
#     output2.write(sbatch)
# print(compile_job_path)


# sbatch = ""
# ct = 0

# for name in folds_arr:
#     ct += 1
#     #for each unique lake
#     print(name)
#     l = name

#     header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp6.out\n#SBATCH --error=EALSTM_%s_oversamp6.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
#     script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
#     script2 = "python EALSTM_err_est_oversamp6.py %s"%(l)
#     all= "\n".join([header,script,script2])
#     sbatch = "\n".join(["sbatch job_%s_foldEAoversamp6.sh"%(l),sbatch])
#     with open('../../hpc/job_{}_foldEAoversamp6.sh'.format(l), 'w') as output:
#         output.write(all)



# compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp6.sh'
# with open(compile_job_path, 'w') as output2:
#     output2.write(sbatch)
# print(compile_job_path)


sbatch = ""
ct = 0

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp7.out\n#SBATCH --error=EALSTM_%s_oversamp7.err\n#SBATCH --gres=gpu:k40:1\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp7.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp7.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp7.sh'.format(l), 'w') as output:
        output.write(all)



compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp7.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)


sbatch = ""
ct = 0

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp8.out\n#SBATCH --error=EALSTM_%s_oversamp8.err\n#SBATCH --gres=gpu:k40:1\n#SBATCH -p k40"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp8.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp8.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp8.sh'.format(l), 'w') as output:
        output.write(all)



compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp8.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)


sbatch = ""
ct = 0

for name in folds_arr:
    ct += 1
    #for each unique lake
    print(name)
    l = name

    header = "#!/bin/bash -l\n#SBATCH --time=23:59:00\n#SBATCH --ntasks=8\n#SBATCH --mem=20g\n#SBATCH --mail-type=ALL\n#SBATCH --mail-user=willa099@umn.edu\n#SBATCH --output=EALSTM_%s_oversamp_norm2.out\n#SBATCH --error=EALSTM_%s_oversamp_norm2.err\n#SBATCH --gres=gpu:v100:1\n#SBATCH -p v100"%(l,l)
    script = "source /home/kumarv/willa099/takeme_evaluate.sh\n" #cd to directory with training script
    script2 = "python EALSTM_err_est_oversamp_norm2.py %s"%(l)
    all= "\n".join([header,script,script2])
    sbatch = "\n".join(["sbatch job_%s_foldEAoversamp_norm2.sh"%(l),sbatch])
    with open('../../hpc/job_{}_foldEAoversamp_norm2.sh'.format(l), 'w') as output:
        output.write(all)



compile_job_path= '../../hpc/sbatch_script_err_est_ea_oversamp_norm2.sh'
with open(compile_job_path, 'w') as output2:
    output2.write(sbatch)
print(compile_job_path)