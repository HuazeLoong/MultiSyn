import os 
import sys
import torch
from itertools import islice
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

class MyTestDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None):
        super(MyTestDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process(xd, xt, xt_featrue, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

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

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    def get_data(self, slice):
        d = [self.data[i] for i in slice]
        return MyTestDataset(d)

    """
    Customize the process method to fit the task of drug-target affinity prediction
    Inputs:
    XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    Y: list of labels (i.e. affinity)
    Return: PyTorch-Geometric format processed data """
    def process(self, xd, xt, xt_featrue, y, graph):
        assert (len(xd) == len(xt) and len(xt) == len(y))
        data_list = []
        slices = [0]
        for i in range(len(xd)):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            cell = self.get_cell_feature(target, xt_featrue)
            if cell is None: # 如果读取cell失败则中断程序
                print('Cell feature not found for target:', cell)
                sys.exit()

            data = DATA.Data()
            new_cell = []
            for n in cell:
                new_cell.append(float(n))
            data.cell = torch.FloatTensor([new_cell])

            data.graph = graph[smiles]
            data.y = torch.Tensor([labels])

            data_list.append(data)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        # print('data',data,'slices',slices)
        torch.save((data, slices), self.processed_paths[0])