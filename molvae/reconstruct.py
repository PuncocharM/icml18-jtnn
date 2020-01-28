import torch
from tqdm import tqdm

from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem

from jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

with open(opts.test_path) as f:
    data = [line.strip("\r\n ").split()[0] for line in f]
        
acc = 0.0
tot = 0
with tqdm(data) as progress_bar:
    for smiles in progress_bar:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        try:
            #dec_smiles = model.reconstruct(smiles3D)
            z = model.encode_latent_mean([smiles3D])[0]
            dec_smiles = model.decode(z[:model.latent_size//2].unsqueeze(0), z[model.latent_size//2:].unsqueeze(0), False)
            
            
            if dec_smiles == smiles3D:
                acc += 1
            tot += 1
            progress_bar.set_postfix(recon_error=acc/tot)
        except Exception as e:
            print(e)

        # dec_smiles = model.recon_eval(smiles3D)
        # tot += len(dec_smiles)
        # for s in dec_smiles:
        #     if s == smiles3D:
        #         acc += 1

print(acc / tot)


