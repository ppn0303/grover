"""
this is randomized search code for grover. newbie make this :L
"""

import random
import torch

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
    init_lr_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    dropout_list = [0, 0.05, 0.1, 0.15, 0.2]
#    attn_hidden_list = [32, 64, 128, 192, 256]
#    attn_out_list = [4, 8]
#    dist_coff_list = [0.05, 0.1, 0.15]
#    bond_drop_rate_list = [0, 0.2, 0.4, 0.6]
    ffn_num_layer_list = [2, 3]
    ffn_dense_list = [120, 150, 180, 200, 220]

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
#        args.attn_out = np.random.choice(attn_out_list, 1)[0]
#        args.dist_coff = np.random.choice(dist_coff_list, 1)[0]
#        args.bond_drop_rate = np.random.choice(bond_drop_rate_list, 1)[0]
        args.ffn_num_layer = np.random.choice(ffn_num_layer_list, 1)[0]
        args.ffn_hidden_size = np.array(np.random.choice(ffn_dense_list, 1)[0])
        params.append(f'\n{iter_num}th search parameter : init_lr is {args.init_lr} \n dropout is {args.dropout} \n attn_out is {args.attn_out} \n dist_coff is {args.dist_coff} \n bond_drop_rate is {args.bond_drop_rate} \n ffn_num_layer is {args.ffn_num_layer} \n ffn_hidden_size is {args.ffn_hidden_size}')
        info(params[iter_num])


        args.seed = init_seed                        # if change this, result will be change
        args.save_dir = os.path.join(save_dir, f'iter_{iter_num}')
        makedirs(args.save_dir)
        if args.parser_name == "finetune":
            model_scores = run_training(args, time_start, logger)
        else:
            model_scores = run_evaluation(args, logger)
        all_scores.append(model_scores)

        # best setting save
        if max(all_scores)==model_scores : 
            best_iter = iter_num
            best_score = model_scores
            best_param = params[iter_num]

    all_scores = np.array(all_scores)

    # Report scores for each iter
    info(f'{args.n_iters}-iter random search')

    for fold_num, scores in enumerate(all_scores):
        info(params[fold_num])
        info(f'Seed {init_seed} ==> test {args.metric} = {np.nanmean(scores):.6f}')

        if args.show_individual_scores:
            for task_name, score in zip(task_names, scores):
                info(f'Seed {init_seed} ==> test {task_name} {args.metric} = {score:.6f}')

    # Report scores across models
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
    info(f'std={std_score:.6f}')
    info(f'best_iter : {best_iter}\nbest_score is {np.nanmean(best_score)}\nbest_param : {best_param}')
    if args.show_individual_scores:
        for task_num, task_name in enumerate(task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')

    return mean_score, std_score