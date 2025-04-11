import os 
import sys
import csv
import torch
import numpy as np
from const import *
from itertools import islice
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

class MyTestDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_feature1=None,xt_feature2=None, transform=None,
                 pre_transform=None, smile_graph=None):
        """
        Initialization function: try to load existing cached data, otherwise execute process to create graph data.

        Parameter description:
        - xd: drug name
        - xt: cell line ID list
        - y: label list
        - xt_feature1: cell line expression feature
        - xt_feature2: cell line fusion feature matrix
        - smile_graph: "SMILES → molecular graph" corresponding mapping (dict)
        """
        self.cell2id = self.load_cell2id(CELL_ID_DIR)   # Load the cell line index mapping table
        self.testcell = np.load(CELL_FEA_DIR)   # Load cell line expression features

        super(MyTestDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
            print('Use existing data files')
        else:
            self.process(xd, xt, xt_feature1,xt_feature2, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
            print('Create a new data file')

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt'] 

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature1(self, cellId, cell_features):
        """ Find the corresponding feature vector in xt_feature1 according to cellId """
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False
    
    def load_cell2id(self, cell2id_file):
        """ 
        Read the cell line to index mapping table (CELL_ID_DIR) 
        and find the location of the corresponding cell line in the fusion feature
        """
        cell2id = {}
        with open(cell2id_file, 'r') as file:
            csv_reader = csv.reader(file, delimiter='\t')
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                cell2id[row[0]] = int(row[1])
        return cell2id    
    def get_cell_feature2(self, cellId):
        """ Get cell line expression characteristics based on cellId """
        if cellId in self.cell2id:
        # if cellId in self.cell2id.values():    
            cell_index = self.cell2id[cellId]
            return self.testcell[cell_index]
        return False

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return MyTestDataset(d)

    """
    Customize the process method to fit the task of drug-target affinity prediction
    Inputs:
    XD - list of DRUG_NAME, XT: list of encoded target (categorical or one-hot),
    Y: list of labels (i.e. affinity)
    Return: PyTorch-Geometric format processed data 
    """
    def process(self, xd, xt, xt_feature1, xt_feature2, y, graph):
        assert (len(xd) == len(xt) and len(xt) == len(y))
        data_list = []
        slices = [0]
        for i in range(len(xd)):
            drug = xd[i]    # drug name
            target = xt[i]  # cell line ID
            labels = y[i]   # label

            # Get cell line expression characteristics 1
            cell1 = self.get_cell_feature1(target, xt_feature1)
            if cell1 is None: 
                print('Cell feature not found for target:', cell1)
                sys.exit()

            # Get cell line fusion PPI feature 2
            cell2 = self.get_cell_feature2(target)
            if cell2 is False: 
                print('Cell feature2 not found for target:', target)
                sys.exit()

            data = DATA.Data()

            # Processing cell features
            new_cell1 = []
            for n in cell1:
                new_cell1.append(float(n))
            data.cell1 = torch.FloatTensor([new_cell1])
            
            if isinstance(cell2, list) and isinstance(cell2[0], np.ndarray):
                new_cell2 = np.array(cell2)
            else:
                new_cell2 = cell2
            if new_cell2.ndim == 1:
                new_cell2 = np.expand_dims(new_cell2, axis=0)
            data.cell2 = torch.FloatTensor(new_cell2)

            data.graph = graph[drug]
            data.y = torch.Tensor([labels])

            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
