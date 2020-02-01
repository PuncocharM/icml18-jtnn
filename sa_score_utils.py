from rdkit import Chem
import pandas as pd
import numpy as np

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
    

def load_targets(tr_scores_path):
    df = pd.read_csv(tr_scores_path)

    tr_means = df.apply(np.mean)
    tr_stds = df.apply(np.std)

    df_normalized = (df - tr_means) / tr_stds

    return df_normalized.values.sum(axis=1)