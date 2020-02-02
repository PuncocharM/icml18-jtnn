from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import networkx as nx
from bo import sascorer

def compute_scores(smiles):
    mol = Chem.MolFromSmiles(smiles)
    logP = Descriptors.MolLogP(mol)
    SA_score = -sascorer.calculateScore(mol)
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    cycle_score = -cycle_length

    return {'SA_score' : SA_score, 'logP' : logP, 'cycle_score' : cycle_score}
    
    
def normalize(score, mean, std):
    return (score - mean) / std

def normalize_scores(scores, means, stds):
    return {k : normalize(scores[k], means[k], stds[k]) for k in scores.keys()}

def compute_normalized_scores(smiles, means, stds):
    return normalize_scores(compute_scores(smiles), means, stds)

def scores_to_target(scores, means, stds):
    return sum(normalize_scores(scores, means, stds).values())

def smiles_to_target(smiles, means, stds):
    return sum(compute_normalized_scores(smiles, means, stds).values())
       
def load_sa_data(tr_scores_path):
    df = pd.read_csv(tr_scores_path)

    tr_means = df.apply(np.mean)
    tr_stds = df.apply(np.std)

    df_normalized = (df - tr_means) / tr_stds

    tr_targets = df_normalized.values.sum(axis=1)
    
    return tr_targets, tr_means, tr_stds


def precompute():
    # Compute TR Scores
    tr_scores = [compute_scores(s) for s in tqdm(tr_smiles)]
    df = pd.DataFrame(tr_scores, columns=['SA_score', 'logP', 'cycle_score'])
    df.to_csv('data/{}/tr_scores.csv'.format(DATASET), index=False)
    with open('data/{}/tr_scores.pic'.format(DATASET), 'wb') as f:
        pickle.dump(tr_scores, f)

    # Compute TR latent vectors
    batch_size = 100
    latent_points = []
    for i in tqdm(list(range(0, len(tr_smiles), batch_size))):
        batch = tr_smiles[i:i+batch_size]
        mol_vec = model.encode_latent_mean(batch)
        latent_points.append(mol_vec.data.cpu().numpy())
    latent_points = np.vstack(latent_points)
    np.savetxt('data/{}/tr_latent.txt'.format(DATASET), latent_points)