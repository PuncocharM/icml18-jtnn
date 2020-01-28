from tqdm import tqdm

import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
if torch.cuda.is_available():
    print('Using CUDA')
    model.load_state_dict(torch.load(args.model))
    model = model.cuda()
else:
    print('Using CPU')
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

torch.manual_seed(0)
for i in tqdm(range(args.nsample)):
    print(rdkit.Chem.MolToSmiles(model.sample_prior()))
