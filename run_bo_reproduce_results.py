import os
#os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'

import theano

print(theano.config.device)

print(theano.config.mode)
print(theano.config.optimizer)
print(theano.config.linker)

from bootstrap_model import bootstrap_model

USE_FAST = False # use the new (faster) version of the code
DATASET = 'zinc'
TRAIN_PATH = 'data/{}/train.txt'.format(DATASET)

model = bootstrap_model(USE_FAST, DATASET)

from sa_score_utils import *

tr_targets, tr_means, tr_stds = load_sa_data('data/{}/tr_scores.csv'.format(DATASET))

X = np.loadtxt('data/{}/tr_latent.txt'.format(DATASET))
y = -tr_targets
y = y.reshape((-1, 1))

n = X.shape[0]

import gzip
import pickle

def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest:
        dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source:
        ret = pickle.load(source, encoding='latin1')
    return ret

from bo.sparse_gp import SparseGP
import scipy.stats as sps
from tqdm import tqdm

M = 500  # default 500
N_EXPERIMENTS = 10 # default 10
N_ITERATIONS = 5  # default 5
MAX_EPOCHS = 100  # default 100
NEXT_N_INPUTS = 60  # default 60
LEARNING_RATE = 0.001  # default 0.001
SAVE_DIR_BASE = 'reproduce-results/'


random_seeds = list(range(1,N_EXPERIMENTS+1))

for random_seed in random_seeds:
    print('-- Experiment %d --' % random_seed)
    SAVE_DIR = SAVE_DIR_BASE + 'experiment-%d/' % random_seed
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    np.random.seed(random_seed)
    permutation = np.random.choice(n, n, replace = False)

    X_train = X[permutation, :][0:np.int(np.round(0.9 * n)), :]
    X_test  = X[permutation, :][np.int(np.round(0.9 * n)):, :]

    y_train = y[permutation][0:np.int(np.round(0.9 * n))]
    y_test  = y[permutation][np.int(np.round(0.9 * n)):]


    for iteration in range(N_ITERATIONS):
        np.random.seed(iteration * random_seed)
        
        print('Iteration %d' % iteration)
        print('  Training the model')
        sgp = SparseGP(X_train, 0*X_train, y_train, M)
        sgp.train_via_ADAM(X_train, 0*X_train, y_train, X_test, X_test*0, y_test, minibatch_size=10*M, max_iterations=MAX_EPOCHS, learning_rate=LEARNING_RATE, verbose=0, progress_bar=tqdm)

        pred, uncert = sgp.predict(X_test, 0 * X_test)
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        print('    Test RMSE: ', error)
        print('    Test ll: ', testll)

        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print('    Train RMSE: ', error)
        print('    Train ll: ', trainll)

        print('  Batched greedy EI')
        next_inputs = sgp.batched_greedy_ei(NEXT_N_INPUTS, np.min(X_train, 0), np.max(X_train, 0), verbose=1, progress_bar=tqdm)
        valid_smiles = []
        new_features = []
        for next_input in next_inputs:
            all_vec = next_input.reshape((1,-1))
            tree_vec, mol_vec = np.hsplit(all_vec, 2)
            s = model.decode_numpy(tree_vec, mol_vec, prob_decode=False)
            if s is not None: 
                valid_smiles.append(s)
                new_features.append(all_vec)

        print('  %d molecules are found' % len(valid_smiles))
        valid_smiles = valid_smiles[:50]
        new_features = new_features[:50]
        new_features = np.vstack(new_features)
        save_object(valid_smiles, SAVE_DIR + "/valid_smiles{}.dat".format(iteration))


        scores = []
        for smile in valid_smiles:
            scores.append(-smiles_to_target(smile, tr_means, tr_stds))

        save_object(scores, SAVE_DIR + "/scores{}.dat".format(iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([X_train, new_features], 0)
            y_train = np.concatenate([y_train, np.array(scores)[:, None]], 0)
