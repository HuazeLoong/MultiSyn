import os

SUB_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(SUB_PROJ_DIR, 'data')

RESULT_DIR = os.path.join(DATA_DIR,'result')
RESULTS_DIR = os.path.join(RESULT_DIR,'results')
LOSS_DIR = os.path.join(RESULT_DIR,'loss')

DRUGID_DIR = os.path.join(DATA_DIR, 'drug_id.csv')
CELL_DIR = os.path.join(DATA_DIR, 'cell_features.csv')
SMILES_DIR = os.path.join(DATA_DIR, 'smiles.csv')

FILE_AUCS_TRAIN = os.path.join(RESULTS_DIR,'train')
FILE_AUCS_TEST = os.path.join(RESULTS_DIR,'test')
LOSSES_DIR = os.path.join(LOSS_DIR,'loss')
