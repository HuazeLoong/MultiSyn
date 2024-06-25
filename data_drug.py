import os
import dgl
import csv
import torch
import numpy as np
import pandas as pd
from const import *
from dataset_drug import *
from itertools import islice
from rdkit import Chem
from rdkit import RDConfig
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.BRICS import FindBRICSBonds


# drug data                                                                                                                                                       
RDLogger.DisableLog('rdApp.*')  

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef') 
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def bond_features(bond: Chem.rdchem.Bond): 
    if bond is None: 
        fbond = [1] + [0] * (BOND_FDIM - 1) # 12
    else:
        bt = bond.GetBondType()
        fbond = [ 
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE, 
            bt == Chem.rdchem.BondType.DOUBLE, 
            bt == Chem.rdchem.BondType.TRIPLE, 
            bt == Chem.rdchem.BondType.AROMATIC, 
            (bond.GetIsConjugated() if bt is not None else 0), # 共轭
            (bond.IsInRing() if bt is not None else 0) # 环
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def pharm_property_types_feats(mol,factory=factory): 
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types) 
    for i in range(len(types)):
        if types[i] in list(set(feats)): 
            result[i] = 1

    return result 

def GetBricsBonds(mol):  
    brics_bonds = list()  
    brics_bonds_rules = list()  
      
    bonds_tmp = FindBRICSBonds(mol)  
    bonds = [b for b in bonds_tmp]  
      
    for item in bonds:  # item[0]是键，item[1]是BRICS类型  
        # 将键的两个原子索引添加到brics_bonds列表中  
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
    return list(MACCSkeys.GenMACCSKeys(mol)) 

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():  
        atom.SetAtomMapNum(atom.GetIdx() + 1)      
    return mol  

def GetFragmentFeats(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0], i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol, break_bonds, addDummies=False) # 切成片段
    frags_idx_lst = Chem.GetMolFrags(tmp) # 提取片段
    # 初始化字典以存储原子与片段的映射以及片段属性
    result_ap = {}
    result_p = {}
    pharm_id = 0
    
    # 遍历片段
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id 
        try:
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, frag_idx)) # 片段转成smiles再转成mol_pharm
            emb_0 = maccskeys_emb(mol_pharm)  
            emb_1 = pharm_property_types_feats(mol_pharm)  
        except Exception:
            emb_0 = [0 for i in range(167)]
            emb_1 = [0 for i in range(27)]       

        result_p[pharm_id] = emb_0 + emb_1
        pharm_id += 1
    
    return result_ap, result_p  # 返回原子与片段的映射,片段特征

ELEMENTS = [35, 6, 7, 8, 9, 15, 16, 17, 53]

ATOM_FEATURES = {
    'atomic_num': ELEMENTS,  
    'degree': [0, 1, 2, 3, 4, 5],  
    'formal_charge': [-1, -2, 1, 2, 0],  # 原子的形式电荷
    'chiral_tag': [0, 1, 2, 3],  # 原子的立体标记
    'num_Hs': [0, 1, 2, 3, 4],  # 与原子相连的氢原子数
    'hybridization': [  # 原子的杂化方式
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1) 
    index = choices.index(value) if value in choices else -1  
    encoding[index] = 1  
    return encoding  

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
    edge_types = [('a','b','a'),('p','r','p'),('a','j','p'), ('p','j','a')]
    edges = {k:[] for k in edge_types}
    
    result_ap, result_p = GetFragmentFeats(mol) # 原子与片段的映射,片段特征
    reac_idx, bbr = GetBricsBonds(mol)
    
    #atom-level 
    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    # pharm-level
    for r in reac_idx:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]]) # 根据原子与片段的映射，将片段与片段连接起来
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]]) # 即 react 连接

    # junction-level
    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    
    f_atom = []
    for idx in g.nodes('a'): 
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(atom_features(atom))
    f_atom = torch.FloatTensor(f_atom) # 框架通常要求输入数据为张量
    g.nodes['a'].data['f'] = f_atom
    # print(g.nodes['a'].data)
    dim_atom = len(f_atom[0])           

    f_pharm = []
    for k,v in result_p.items(): 
        f_pharm.append(v)
    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]

    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)
    
    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))  # beginnode, endnode
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    f_reac = []
    src, dst = g.edges(etype=('p','r','p'))
    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        for i in bbr: # bbr BrICS-Bond
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1: #原子节点需要和片段节点（原子节点映射）对应 
                f_reac.append(i[1])
    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)

    return g

# cell data
def get_cell_feature(cellId, cell_features):
    for row in islice(cell_features, 0, None):
        if row[0] == cellId:
            return row[1: ]

def creat_data(datafile,cellfile,smilefile):
    cell_features = []
    with open(cellfile) as file:
        csv_reader = csv.reader(file)  
        for row in csv_reader:
            cell_features.append(row)
    cell_features = np.array(cell_features)

    compound_iso_smiles = []
    df = pd.read_csv(smilefile)
    compound_iso_smiles += list(df['smile'])
    compound_iso_smiles = set(compound_iso_smiles)
    
    smile_pharm_graph = {}
    for smile in compound_iso_smiles:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HeteroGraph(mol)
        smile_pharm_graph[smile] = g

    data_file = os.path.join(DATA_DIR, datafile)
    df = pd.read_csv(data_file + '.csv')
    drug1, drug2, cell, label = list(df['drug1']), list(df['drug2']), list(df['cell']), list(df['label'])
    drug1, drug2, cell, label = np.asarray(drug1), np.asarray(drug2), np.asarray(cell), np.asarray(label)

    MyTestDataset(root=DATA_DIR, dataset=datafile + '_drug1', xd=drug1, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_pharm_graph)
    MyTestDataset(root=DATA_DIR, dataset=datafile + '_drug2', xd=drug2, xt=cell, xt_featrue=cell_features, y=label,smile_graph=smile_pharm_graph)

if __name__ == "__main__":
    datafile_1 = ['new_labels_0_10']
    for datafile in datafile_1:
        creat_data(datafile, CELL_DIR, SMILES_DIR)
