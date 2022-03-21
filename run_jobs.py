import os
from time import sleep

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

maze = 'Point4Rooms-v1'
nb_seeds = 5
values = [0., 1., 0.01, 0.1]

for i in range(nb_seeds):
    for value in values:
        job_file = os.path.join(job_directory, "sp_{}%.slurm".format(value))

        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --account=kcr@gpu\n")
            fh.writelines("#SBATCH --job-name={}_sp_{}\n".format(maze, value))
            fh.writelines("#SBATCH --qos=qos_gpu-t3\n")
            fh.writelines("#SBATCH --output={}_sp_{}%_%j.out\n".format(maze, value))
            fh.writelines("#SBATCH --error={}_sp_{}%_%j.out\n".format(maze, value))
            fh.writelines("#SBATCH --time=9:59:59\n")
            fh.writelines("#SBATCH --ntasks=8\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --gres=gpu:1\n")
            fh.writelines("#SBATCH --hint=nomultithread\n")
            fh.writelines("#SBATCH --array=0-0\n")

            fh.writelines("module load pytorch-gpu/py3/1.4.0\n")

            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")
            fh.writelines("export OMPI_MCA_opal_warn_on_missing_libcuda=0\n")
            fh.writelines("export OMPI_MCA_btl_openib_allow_ib=1\n")
            fh.writelines("export OMPI_MCA_btl_openib_warn_default_gid_prefix=0\n")
            fh.writelines("export OMPI_MCA_mpi_warn_on_fork=0\n")

            fh.writelines("srun python -u -B train.py --n-epochs 200  --env-name {} --n-cycles 10 --n-batches 30 --save-freq 2 --intervention-prob {} --save-dir '{}_sp_{}/' 2>&1 ".format(maze, value, maze, value))

        os.system("sbatch %s" % job_file)
        sleep(1)