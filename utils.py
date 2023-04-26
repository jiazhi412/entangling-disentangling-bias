import os
import json
import time
import logging
import torch
import pandas as pd
from itertools import product

def append_data_to_csv(data,csv_name):
    df = pd.DataFrame(data)
    if os.path.exists(csv_name):
        df.to_csv(csv_name,mode='a',index=False,header=False)
    else:
        df.to_csv(csv_name,index=False)

def compute_subAcc_withlogits_binary(logits, target, a):
    # output is logits and predict_prob is probability
    assert logits.shape == target.shape, f"Acc, output {logits.shape} and target {target.shape} are not matched!"
    predict_prob = torch.sigmoid(logits)
    predict_prob = predict_prob.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    a = a.cpu().detach().numpy()

    # Young
    tmp = a <= 0
    predict_prob_n = predict_prob[tmp.nonzero()]
    target_n = target[tmp.nonzero()]
    Acc_n = (predict_prob_n.round() == target_n).mean()
    tmp = a > 0
    predict_prob_p = predict_prob[tmp.nonzero()]
    target_p = target[tmp.nonzero()]
    Acc_p = (predict_prob_p.round() == target_p).mean()
    # print(target)
    # print(tmp)
    # print(predict_prob_p.shape)
    # print(predict_prob_n.shape)
    # print('jdasljd')
    return Acc_p, Acc_n


def run(command_template, qos, gpu, *args):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # if not os.path.exists('errors'):
    #     os.makedirs('errors')

    l = len(args)
    job_name_template = '{}'
    for _ in range(l-1):
        job_name_template += '-{}'
    for a in product(*args):
        command = command_template.format(*a)
        job_name = job_name_template.format(*a)
        bash_file = '{}.sh'.format(job_name)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('#!/bin/bash\n')
            OUT.write('#SBATCH --job-name={} \n'.format(job_name))
            OUT.write('#SBATCH --ntasks=1 \n')
            OUT.write('#SBATCH --account=other \n')
            OUT.write(f'#SBATCH --qos={qos} \n')
            OUT.write('#SBATCH --partition=ALL \n')
            OUT.write('#SBATCH --cpus-per-task=4 \n')
            OUT.write(f'#SBATCH --gres=gpu:{gpu} \n')
            OUT.write('#SBATCH --mem={}G \n'.format(16 * gpu))
            OUT.write('#SBATCH --time=5-00:00:00 \n')
            OUT.write('#SBATCH --exclude=vista[03] \n')
            OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
            OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
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
