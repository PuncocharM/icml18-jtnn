import torch

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def bootstrap_model(use_fast=False, dataset='zinc'):
    if use_fast:
        from fast_jtnn import JTNNVAE, Vocab, create_var
    else:
        from jtnn import JTNNVAE, Vocab, create_var

    VOCAB = 'data/{}/vocab.txt'.format(dataset)
    TRAIN_PATH = 'data/{}/train.txt'.format(dataset)
    TEST_PATH = 'data/{}/test.txt'.format(dataset)
    HIDDEN_SIZE = 450
    LATENT_SIZE = 56
    if use_fast:
        DEPTH_T = 20
        DEPTH_G = 3
    else:
        DEPTH = 3
    STEREO = True
    if use_fast and dataset=='moses':
        MODEL = 'fast_molvae/moses-h450z56/model.iter-400000'
    elif not use_fast and dataset=='moses':
        MODEL = 'molvae/moses-h450L56d3beta0.5/model.iter-2'
    elif not use_fast and dataset=='zinc':
        MODEL = 'molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4'
    else:
        raise Exception('invalid use_fast and dataset combination')
    
    vocab = [x.strip("\r\n ") for x in open(VOCAB)] 
    vocab = Vocab(vocab)

    if use_fast:
        model = JTNNVAE(vocab, HIDDEN_SIZE, LATENT_SIZE, DEPTH_T, DEPTH_G)
    else:
        model = JTNNVAE(vocab, HIDDEN_SIZE, LATENT_SIZE, DEPTH, STEREO)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.load_state_dict(torch.load(MODEL))
        model = model.cuda()
    else:
        print('Using CPU')
        model.load_state_dict(torch.load(MODEL, map_location='cpu'))
        
    return model