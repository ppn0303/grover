"""
The cross validation function for finetuning.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/train/cross_validate.py
"""
import os
import time
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np

from grover.util.utils import get_task_names
from grover.util.utils import makedirs
from task.run_evaluation import run_evaluation
from task.train import run_training

import random
import torch

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    k-fold cross validation.

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    # Run training with different random seeds for each fold
    all_scores = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        if args.parser_name == "finetune":
            model_scores = run_training(args, time_start, logger)
        else:
            model_scores = run_evaluation(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    info(f'{args.num_folds}-fold cross validation')

    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
    info(f'std={std_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parametersearch(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    randomsearch

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    #randomize parameter list
    init_lr_list = [0.00001, 0.00003, 0.00005] #[0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009, 0.0001]
    dropout_list = [0, 0.05, 0.1, 0.15, 0.2]
#    attn_hidden_list = [32, 64, 128, 192, 256]      #in paper this is 128 hold
    attn_out_list = [4, 8]
    dist_coff_list = [0.05, 0.1, 0.15]
    bond_drop_rate_list = [0, 0.2, 0.4, 0.6]
    ffn_num_layer_list = [2, 3, 4]
    ffn_dense_list = [300, 400, 500, 600, 700]

    # Run training with different random seeds for each fold
    all_scores = []
    params = []
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    for iter_num in range(0, args.n_iters):
        info(f'iter {iter_num}')

        #randomize parameter
        np.random.seed()
        random.seed()
        args.init_lr = np.random.choice(init_lr_list, 1)[0]
        args.max_lr = args.init_lr * 10
        args.final_lr = args.init_lr
        args.dropout = np.random.choice(dropout_list, 1)[0]
        args.attn_out = np.random.choice(attn_out_list, 1)[0]
        args.dist_coff = np.random.choice(dist_coff_list, 1)[0]
        args.bond_drop_rate = np.random.choice(bond_drop_rate_list, 1)[0]
        args.ffn_num_layer = np.random.choice(ffn_num_layer_list, 1)[0]
        args.ffn_hidden_size = np.array(np.random.choice(ffn_dense_list, 1)[0])
        params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layer is {args.ffn_num_layer} \n ffn_hidden_size is {args.ffn_hidden_size}')
        info(params[iter_num])

        args.seed = init_seed                        # if change this, result will be change
        iter_dir = os.path.join(save_dir, f'iter_{iter_num}')
        args.save_dir = iter_dir
        makedirs(args.save_dir)

        fold_scores = []
        for fold_num in range(args.num_folds):
            info(f'Fold {fold_num}')
            args.seed = init_seed + fold_num
            args.save_dir = os.path.join(iter_dir, f'fold_{fold_num}')
            makedirs(args.save_dir)
            if args.parser_name == "finetune":
                model_scores = run_training(args, time_start, logger)
            else:
                model_scores = run_evaluation(args, logger)
#change below line for compare average score
            fold_scores.append(model_scores)
        fold_scores = np.array(fold_scores)

        # Report scores for each fold
        info(f'\n{args.num_folds}-fold validation')
        info(f'{params[iter_num]}\n')

        for fold_num, scores in enumerate(fold_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        fold_avg_scores = np.nanmean(fold_scores, axis=1)  # average score for each model across tasks
        fold_mean_score, fold_std_score = np.nanmean(fold_avg_scores), np.nanstd(fold_avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={fold_mean_score:.6f}')
        info(f'std={fold_std_score:.6f}\n')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(fold_scores[:, task_num]):.6f} +/- {np.nanstd(fold_scores[:, task_num]):.6f}')

        all_scores.append(fold_mean_score)

############fold end, save fold_data and initialize seed

        # best setting save
        if max(all_scores)==fold_mean_score : 
            best_iter = iter_num
            best_score = fold_mean_score
            best_param = params[iter_num]

############iter end

    all_scores = np.array(all_scores)

    # Report scores for each iter
    info(f'\n---- {args.n_iters}-iter random search ----')

    for iter_num, scores in enumerate(all_scores):
        info(params[iter_num])
        info(f'Seed {init_seed} ==> test {args.metric} = {np.nanmean(scores):.6f}\n')

    # Report best model
    info(f'\nbest_iter : {best_iter}\nbest_score is {np.nanmean(best_score)}\nbest_param : {best_param}')

    return best_score
