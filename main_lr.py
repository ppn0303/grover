import random

import numpy as np
import torch
from rdkit import RDLogger

from grover.util.parsing import parse_args, get_newest_train_args
from grover.util.utils import create_logger
from task.cross_validate import cross_validate #, grid_search
from task.fingerprint import generate_fingerprints
from task.predict import make_predictions, write_prediction
from task.pretrain import pretrain_model
from grover.data.torchvocab import MolVocab

#add for gridsearch
from sklearn.model_selection import GridSearchCV
from argparse import ArgumentParser, Namespace

def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # setup random seed
    setup(seed=42)
    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args()
    if args.parser_name == 'finetune':
        for i in (0.00001,0.00005):
            args.init_lr = i
            args.final_lr = i
            args.max_lr = i*10
            if i == 0.00001 : args.save_dir = 'model/tox21_150_lr000001' 
            else : args.save_dir = 'model/tox21_150_lr000005'
            print(f'lr is {args.init_lr}')
            logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
            cross_validate(args, logger)
    elif args.parser_name == 'pretrain':
        logger = create_logger(name='pretrain', save_dir=args.save_dir)
        pretrain_model(args, logger)
    elif args.parser_name == "eval":
        logger = create_logger(name='eval', save_dir=args.save_dir, quiet=False)
        cross_validate(args, logger)
    elif args.parser_name == 'fingerprint':
        train_args = get_newest_train_args()
        logger = create_logger(name='fingerprint', save_dir=None, quiet=False)
        feas = generate_fingerprints(args, logger)
        np.savez_compressed(args.output_path, fps=feas)
    elif args.parser_name == 'predict':
        train_args = get_newest_train_args()
        avg_preds, test_smiles = make_predictions(args, train_args)
        write_prediction(avg_preds, test_smiles, args)
    elif args.parser_name == 'gridsearch':
        logger = create_logger(name='GridSearch', save_dir=args.save_dir)
        pretrain_model(args, logger)

        

