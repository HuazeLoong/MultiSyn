import os
import dgl
import csv
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.BRICS import FindBRICSBonds

from const import *
from dataset import *

"""
This script handles the preprocessing and molecular graph construction for drug combination prediction. 
It parses SMILES strings, generates heterogeneous graphs with atom and pharmacophore nodes, and saves processed datasets compatible.

Key functionalities:
- Fragmentation of molecules using BRICS
- Pharmacophoric node feature extraction
- Mapping between atoms and fragments
- Construction of heterogeneous graphs
- Dataset construction and saving

Dependencies: RDKit, DGL, Torch, Pandas
"""                           

RDLogger.DisableLog('rdApp.*')  

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef') 
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def bond_features(bond: Chem.rdchem.Bond): 
    """ Extract the feature vector of chemical bonds in RDKit. """
    if bond is None: 
        fbond = [1] + [0] * (BOND_FDIM - 1) # type: ignore
    else:
        bt = bond.GetBondType()
        fbond = [ 
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE, # single bond
            bt == Chem.rdchem.BondType.DOUBLE, # double bond
            bt == Chem.rdchem.BondType.TRIPLE, # triple bond
            bt == Chem.rdchem.BondType.AROMATIC, # aromatic bond
            (bond.GetIsConjugated() if bt is not None else 0), # conjugated bond
            (bond.IsInRing() if bt is not None else 0) # ring
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    """ Generate a binary vector indicating the presence of different pharmacophore property types in a given molecule using the RDKit feature factory. """
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types) 
    for i in range(len(types)):
        if types[i] in list(set(feats)): 
            result[i] = 1

    return result 

def GetBricsBonds(mol):  
    """
    Identify BRICS bonds in a molecule and assign corresponding BRICS rule-based features:
    1. A list of directed BRICS bond pairs (both directions),
    2. A list of BRICS bond rule-based features for each bond direction.
    """
    brics_bonds = list()  
    brics_bonds_rules = list()  
      
    bonds_tmp = FindBRICSBonds(mol)  
    bonds = [b for b in bonds_tmp]  
      
    # Convert each bond to bidirectional mapping with BRICS features
    for item in bonds:  # item[0] is a key, item[1] is a BRICS type 
        brics_bonds.append([int(item[0][0]), int(item[0][1])])  
        brics_bonds_rules.append([[int(item[0][0]), int(item[0][1])], GetBricsBondFeature([item[1][0], item[1][1]])])  
        brics_bonds.append([int(item[0][1]), int(item[0][0])])  
        brics_bonds_rules.append([[int(item[0][1]), int(item[0][0])], GetBricsBondFeature([item[1][1], item[1][0]])])  
  
    result = []  
    for bond in mol.GetBonds():  
        beginatom = bond.GetBeginAtomIdx()  
        endatom = bond.GetEndAtomIdx()  
        if [beginatom, endatom] in brics_bonds:  
            result.append([bond.GetIdx(), beginatom, endatom])  
              
    return result, brics_bonds_rules 

def GetBricsBondFeature(action):  
    """ Generate a one-hot feature vector for a BRICS bond. """
    result = []   
    start_action_bond = int(action[0]) if (action[0] != '7a' and action[0] != '7b') else 7   
    end_action_bond = int(action[1]) if (action[1] != '7a' and action[1] != '7b') else 7   
    emb_0 = [0 for i in range(17)]      
    emb_1 = [0 for i in range(17)]    
    emb_0[start_action_bond] = 1    
    emb_1[end_action_bond] = 1  
    result = emb_0 + emb_1  
    return result

def maccskeys_emb(mol):  
    """Generate a MACCS fingerprint for the given molecule."""
    return list(MACCSkeys.GenMACCSKeys(mol)) 

def mol_with_atom_index(mol):
    """Set the mapping number of the atom according to the index of the atom"""
    for atom in mol.GetAtoms():  
        atom.SetAtomMapNum(atom.GetIdx() + 1)      
    return mol  

def GetFragmentFeats(mol):
    """Break bonds and split molecules based on BRICS rules, extract fragment features and atom-fragment mapping"""
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False) # Cut into segments

    frags_idx_lst = Chem.GetMolFrags(tmp) # Extract fragments

    # Initialize dictionary to store (mappings of atoms to fragments) and (fragment attributes)
    result_ap = {}
    result_p = {}
    pharm_id = 0
    
    # Iterate through the fragments
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id 
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx)) 
            emb_0 = maccskeys_emb(mol_pharm)  
            emb_1 = pharm_property_types_feats(mol_pharm)  
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]       

        result_p[pharm_id] = emb_0 + emb_1
        pharm_id += 1
    
    return result_ap, result_p # Returns the mapping between atoms and fragments, fragment features

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1) 
    index = choices.index(value) if value in choices else -1  
    encoding[index] = 1  
    return encoding  

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]

ATOM_FEATURES = {
    'atomic_num': ELEMENTS,  
    'degree': [0, 1, 2, 3, 4, 5],  
    'formal_charge': [-1, -2, 1, 2, 0],  # Formal charge of the atom
    'chiral_tag': [0, 1, 2, 3],  # Stereo labels of atoms
    'num_Hs': [0, 1, 2, 3, 4],  # Number of hydrogen atoms attached to the atom
    'hybridization': [  # Hybridization of atoms
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01] # scaled to about the same range as other features

    return features  

def Mol2HeteroGraph(mol):
    """
    Converts an RDKit molecule object to a heterogeneous graph.
    The graph contains four edge types: interatomic bonds, interfragment connections, and cross-connections between atoms and fragments.
    Returns the constructed DGL heterogeneous graph object.
    """
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]
    edges = {k:[] for k in edge_types}
    
    result_ap, result_p = GetFragmentFeats(mol)
    reac_idx, bbr = GetBricsBonds(mol) 
    # BRICS bond pairs and their related feature information. 
    # [[start atom index, end atom index], BRICS bond pair feature information].
    
    #atom-level 
    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    # pharm-level
    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    # junction-level
    for k,v in result_ap.items(): # atom_id,pharm_id 
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    
    # add atomic node features
    f_atom = []
    for idx in g.nodes('a'): 
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom)
    g.nodes['a'].data['f'] = f_atom
    # print(g.nodes['a'].data)
    dim_atom = len(f_atom[0])

    # add fragment node features
    f_pharm = []
    for k,v in result_p.items(): 
        f_pharm.append(v)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    # add atomic level edge features (type of bond)
    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))  # beginnode, endnode
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    # add segment-level edge features (BRICS reaction pair information)
    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr: # bbr BrICS-Bond [[start atom index, end atom index], BRICS bond pair feature information]
            p0 = result_ap[i[0][0]] 
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)

    return g

def creat_data(datafile,cellfile1, cellfile2):
    """
    Load cell line features and SMILES molecular data, build drug isomerism graph, and save in MyTestDataset format.
    Input:
    - datafile: drug combination file name (without .csv)
    - cellfile1: cell line simple feature file (CSV format)
    - cellfile2: cell line fusion feature file (npy format)
    """
    cell_features = []
    with open(cellfile1) as file:
        csv_reader = csv.reader(file)  
        for row in csv_reader:
            cell_features.append(row)
    cell_features1 = np.array(cell_features) 

    cell2 = np.load(cellfile2)
    cell_features2 = np.array(cell2)

    # read drug combination data and drug ID comparison table
    data_file = os.path.join(DATA_DIR, datafile)
    df = pd.read_csv(data_file + '.csv')
    df_id = pd.read_csv(DRUGID_DIR)

    compound_iso_smiles = []
    compound_iso_smiles += list(df['drug1_smiles'])
    compound_iso_smiles += list(df['drug2_smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    
    smile_pharm_graph = {}
    for smile in compound_iso_smiles:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HeteroGraph(mol)
        for index, row in df_id.iterrows():
            if row['smiles'] == smile:
                smile_pharm_graph[row['drug']] = g
        
    drug1, drug2, cell, label = list(df['drug1_name']), list(df['drug2_name']), list(df['cell']), list(df['label'])
    drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)

    MyTestDataset(root=DATAS_DIR, dataset=datafile + '_drug1', xd=drug1, xt=cell, xt_feature1=cell_features1, xt_feature2=cell_features2, y=label, smile_graph=smile_pharm_graph)
    MyTestDataset(root=DATAS_DIR, dataset=datafile + '_drug2', xd=drug2, xt=cell, xt_feature1=cell_features1, xt_feature2=cell_features2, y=label, smile_graph=smile_pharm_graph)

if __name__ == "__main__":
    datafile_1 = ['drugcom_12415', 'drugcom_13243']
    for datafile in datafile_1:
        creat_data(datafile,CELL_DIR, CELL_FEA_DIR)
