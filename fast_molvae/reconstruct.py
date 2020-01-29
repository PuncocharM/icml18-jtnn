import torch
from tqdm import tqdm

import argparse

import rdkit
import rdkit.Chem as Chem

from fast_jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument("--model")
parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)
args = parser.parse_args()

print(args)
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))
model = model.cuda()

with open(args.test) as f:
    data = [line.strip("\r\n ").split()[0] for line in f]

smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True) for s in data]

acc = 0.0
tot = 0
with tqdm(list(smiles)) as progress_bar:
    for s in progress_bar:
        try:
            z_mean, z_log_var = model.smiles_to_latent([s])
            z = z_mean[0]
            dec_smiles = model.latent_to_smiles_one(z)
            dec_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(dec_smiles), isomericSmiles=True)

            if dec_smiles == s:
                acc += 1
            tot += 1
            progress_bar.set_postfix(recon_error=acc/tot)
        except Exception as e:
            print(e)


print(acc / tot)


