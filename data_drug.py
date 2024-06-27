from model_drug import *

class TestSyn(nn.Module):
    def __init__(self, n_output=2, num_features_xt=954, dropout=0.2, output_dim=128):
        super(TestSyn, self).__init__()
        hid_dim = 300  
        self.act = get_func("ReLU")  
        self.depth = 3  

        self.w_atom = nn.Linear(42, hid_dim)
        self.w_bond = nn.Linear(14, hid_dim)

        self.w_pharm = nn.Linear(194, hid_dim)
        self.w_reac = nn.Linear(34, hid_dim)

        self.w_junc = nn.Linear(42 + 194, hid_dim)

        self.mp = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth, view='a', suffix='h', act=self.act)
        self.mp_aug = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth, view='ap', suffix='aug', act=self.act)
        
        self.readout = Node_GRU(hid_dim)
        self.readout_attn = Node_GRU(hid_dim)

        self.initialize_weights()

        # cell features MLP
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )

        # combined layers
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(hid_dim * 8 + output_dim, 1024),  
            BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512), 
            BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256), 
            BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128), 
            BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_output), 
        )


    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:  
                nn.init.constant_(param, 0)  
            else:
                nn.init.xavier_normal_(param)  

    def init_feature(self, bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a', 'b', 'a')].data['x'] = self.act(self.w_bond(bg.edges[('a', 'b', 'a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        bg.edges[('p', 'r', 'p')].data['x'] = self.act(self.w_reac(bg.edges[('p', 'r', 'p')].data['x']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))
        
    def forward(self, data1, data2):
        # drug_a 
        graph_a_list = data1.graph  
        graph_a = dgl.batch(graph_a_list)  
        self.init_feature(graph_a)  
        self.mp(graph_a) 
        self.mp_aug(graph_a)  
        embed_f_a = self.readout(graph_a, 'h')  
        embed_aug_a = self.readout_attn(graph_a, 'aug') 
        embed_a = torch.cat([embed_f_a, embed_aug_a], 1) 

        # drug_b
        graph_b_list = data2.graph  
        graph_b = dgl.batch(graph_b_list)  
        self.init_feature(graph_b)  
        self.mp(graph_b) 
        self.mp_aug(graph_b)  
        embed_f_b = self.readout(graph_b, 'h')  
        embed_aug_b = self.readout_attn(graph_b, 'aug')  
        embed_b = torch.cat([embed_f_b, embed_aug_b], 1)  

        # cell features
        cell = F.normalize(data1.cell, 2, 1)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((embed_a, embed_b, cell_vector), 1)  
        xc = F.normalize(xc, 2, 1)
        out = self.pred(xc)
        return out
