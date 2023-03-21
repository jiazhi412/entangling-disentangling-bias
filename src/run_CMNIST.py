from __future__ import print_function
import os
from itertools import product
from re import escape

if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('errors'):
    os.makedirs('errors')

# parameters
command_template = 'python EnD_rebias_mnist_sweep_our.py --color_var {} --epochs {}'
# p2 = [0.001, 0.002]
# p2 = [0.001, 0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009]
# p2 = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
p2 = [0.001, 0.002, 0.003, 0.004, 0.006, 0.007, 0.008, 0.009, 
        0.011, 0.012, 0.013, 0.014, 0.016, 0.017, 0.018, 0.019,
      0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
p3 = [80]
# p4 = [10]
# p5 = [1]

for p2, p3 in product(p2, p3):
    command = command_template.format(p2, p3)
    job_name = f'End-{p2}-{p3}'
    bash_file = '{}.sh'.format(job_name)
    with open( bash_file, 'w' ) as OUT:
        OUT.write('#!/bin/bash\n')
        OUT.write('#SBATCH --job-name={} \n'.format(job_name))
        OUT.write('#SBATCH --ntasks=1 \n')
        OUT.write('#SBATCH --account=other \n')
        OUT.write('#SBATCH --qos=premium \n')
        OUT.write('#SBATCH --partition=ALL \n')
        OUT.write('#SBATCH --cpus-per-task=4 \n')
        OUT.write('#SBATCH --gres=gpu:1 \n')
        OUT.write('#SBATCH --mem=16G \n')
        OUT.write('#SBATCH --time=5-00:00:00 \n')
        OUT.write('#SBATCH --exclude=vista[02,05,18] \n')
        # OUT.write('#SBATCH --exclude=vista[06,07,10,11,13,17-20] \n')
        OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
        OUT.write('#SBATCH --error=errors/{}.out \n'.format(job_name))
        OUT.write('source ~/.bashrc\n')
        OUT.write('echo $HOSTNAME\n')
        OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
        OUT.write('conda activate pytorch\n')
        OUT.write(command)

    qsub_command = 'sbatch {}'.format(bash_file)
    os.system( qsub_command )
    os.system('rm -f {}'.format(bash_file))
    print( qsub_command )
    print( 'Submitted' )
