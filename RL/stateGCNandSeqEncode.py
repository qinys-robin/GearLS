'''
Author: Qin Yusen
email: qinys2001@163.com
LastEditTime: 2025-07-10 09:45:52
Description: Modified from Animesh Basak Chowdhury's work ABC_RL(ICLR'24)
             All nueral networks included in RL
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool,SAGPooling,TopKPooling,ASAPooling,global_add_pool
from torch_geometric.nn.norm import BatchNorm,GraphNorm,LayerNorm,InstanceNorm
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch       
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  

allowable_features = {
    'node_type' : [0,1,2],
    'num_inverted_predecessors' : [0,1,2]
}

def get_node_feature_dims():
    return list(map(len, [
        allowable_features['node_type']
    ]))

def numpyToTorch(input_arr, device):
    if isinstance(input_arr, np.ndarray):
        converted = torch.from_numpy(input_arr).to(device)
        return converted
    elif isinstance(input_arr, torch.Tensor):
        return input_arr.to(device)
    else:
        raise TypeError


full_node_feature_dims = get_node_feature_dims()

'''
Node Feature handling:
7-dim --> Linear --> 32-dim
'''
class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_type_embedding = torch.nn.Linear(7, emb_dim)
        torch.nn.init.xavier_uniform_(self.node_type_embedding.weight.data)

    def forward(self, x):
        # First feature is node type, second feature is inverted predecessor
        device = next(self.parameters()).device
        x = numpyToTorch(x, device)
        x_embedding = self.node_type_embedding(x)
        return x_embedding

'''
AIG Graph feature handling:
32-dim node feat --> 3-Layer GCN --> Max and Mean Pooling
Concat pooling output, 64-dim graph embedding
'''
class AIGEncoder(torch.nn.Module):

    def __init__(self,node_encoder,input_dim,num_layer = 3,emb_dim = 32,
                 norm_type='batch',final_layer_readout=True,pooling_type=None,
                 pooling_ratio=0.8):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(AIGEncoder,self).__init__()
        self.num_layer = num_layer
        self.node_emb_size = input_dim
        self.node_encoder = node_encoder
        self.gnn_conv = GCNConv
        self.norm_type = BatchNorm
        self.isPooling = False if pooling_type == None else True
        self.pooling_ratio = pooling_ratio
        self.final_layer_readout = final_layer_readout
        self.readout_max = global_max_pool
        self.readout_mean = global_mean_pool
        
        ### Select the type of Graph Conv Networks
        # if gnn_type == 'gin':
        #     self.gnn_conv = GINConv
        # elif gnn_type == 'gat':
        #     self.gnn_conv = GATConv
        # elif gnn_type == 'tag':
        #     self.gnn_conv = TAGConv
            
        ### Select the type of Normalization
        if norm_type == 'graph':
            self.norm_type = GraphNorm
        elif norm_type == 'layer':
            self.norm_type = LayerNorm
        elif norm_type == 'instance':
            self.norm_type = InstanceNorm
            
        ## Pooling Layers
        if pooling_type == 'topk':
           self.pool_type = TopKPooling
        elif pooling_type == 'sag':
            self.pool_type = SAGPooling
        elif pooling_type == 'asap':
            self.pool_type = ASAPooling

        ###List of GNNs and layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        if self.isPooling:
            self.pools = torch.nn.ModuleList()

        ## First layer
        self.convs.append(self.gnn_conv(input_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))
        if self.isPooling:
            self.pools.append(self.pool_type(emb_dim))

        ## Intermediate Layers
        for _ in range(1, num_layer-1):
            self.convs.append(self.gnn_conv(emb_dim, emb_dim))
            self.norms.append(self.norm_type(emb_dim))
            if self.isPooling:
                self.pools.append(self.pool_type(in_channels=emb_dim,ratio=self.pooling_ratio))
            
        ## Last Layer
        self.convs.append(self.gnn_conv(emb_dim, emb_dim))
        self.norms.append(self.norm_type(emb_dim))
        
        
        ## Global Readout Layers
        # self.readout = []
        # for readoutConfig in readout_type:
        #     if readoutConfig == 'max':
        #         self.readout.append(global_max_pool)
        #     elif readoutConfig == 'mean':
        #         self.readout.append(global_mean_pool)
        #     elif readoutConfig == 'sum':
        #         self.readout.append(global_add_pool)
    
    def forward(self, graph_data:Data | Batch):
        #batch = batched_data.batch
        if isinstance(graph_data, Batch):
            batch = graph_data.batch
        else:
            batch = None
        x, edge_index = graph_data.x, graph_data.edge_index
        #print(x.device, batch.device if batch is not None else ' ')

        #x = torch.cat([batched_data.node_type.reshape(-1, 1),batched_data.num_inverted_predecessors.reshape(-1, 1)], dim=1)
        h = self.node_encoder(x)
        device = next(self.parameters()).device
        edge_index = numpyToTorch(edge_index, device)
        
        finalReadouts = []

        for layer in range(self.num_layer):
            #h = self.convs[layer](h, edge_index)
            h = self.convs[layer](h, edge_index)
            #h = self.norms[layer](h)
            #h = self.norms[layer](h)
            if layer != self.num_layer - 1:
                h = F.relu(h)
                if self.isPooling:                    # Not pooling in the last layer
                    poolOutput = self.pools[layer](h,edge_index=edge_index,batch=batch)
                    h,edge_index,batch = poolOutput[0],poolOutput[1],poolOutput[3]
                if self.final_layer_readout:
                    continue
            
            finalReadouts.append(self.readout_max(h,batch))
            finalReadouts.append(self.readout_mean(h,batch))
        aigEmbedding = torch.cat(finalReadouts,dim=1)
        aigEmbedding = torch.round(aigEmbedding,decimals=3)
        return aigEmbedding
    
class policyNetwork(nn.Module):
    def __init__(self, state_embed_dim = 256, hidden_dim = 256, n_actions = 9):
        super(policyNetwork, self).__init__()
        self.state_embed_dim = state_embed_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.dense_p1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dense_p2 = nn.Linear(self.hidden_dim, self.n_actions)
        torch.nn.init.xavier_uniform_(self.dense_p2.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_p1.weight.data)
        # Google Brain: What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study
        # Multiply the last policy layer weights by 1e-2
        self.dense_p2.weight.data = self.dense_p2.weight.data*0.01

    def forward(self, state_embedding):
        p1Out = F.leaky_relu(self.dense_p1(state_embedding))
        logits = self.dense_p2(p1Out)
        policy = F.softmax(logits, dim=1)
        return logits, policy
    
class valueNetwork(nn.Module):
    def __init__(self, state_embed_dim = 256, hidden_dim = 256):
        super(valueNetwork, self).__init__()
        self.state_embed_dim = state_embed_dim
        self.hidden_dim = hidden_dim

        self.dense_v1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dense_v2 = nn.Linear(self.hidden_dim, 1)
        torch.nn.init.kaiming_uniform_(self.dense_v1.weight.data)
        torch.nn.init.kaiming_uniform_(self.dense_v2.weight.data)

    def forward(self, state_embedding):
        v1Out = F.leaky_relu(self.dense_v1(state_embedding))
        value = torch.tanh(self.dense_v2(v1Out)).view(-1)
        return value

class RLNetwork(nn.Module): 
    def __init__(self, init_graph_data=True,node_enc_outdim=32, gnn_hidden_dim = 32,num_gcn_layer = 3,
                gnn_type = 'gcn',norm_type='batch',final_layer_readout=True, glob_feature_dim = 8,
                pooling_type=None,pooling_ratio=0.8,readout_type=['mean','max'], n_hidden=256,n_actions=9):
        
        super(RLNetwork, self).__init__()
        self.init_graph_data = init_graph_data
        if self.init_graph_data:
            self.node_encoder = NodeEncoder(emb_dim=node_enc_outdim)
            self.aig = AIGEncoder(self.node_encoder,input_dim=node_enc_outdim,num_layer=num_gcn_layer,emb_dim=gnn_hidden_dim,
                                 norm_type=norm_type,final_layer_readout=final_layer_readout,pooling_type=pooling_type,pooling_ratio=pooling_ratio)
            
            #Readout happening after each GCN layer
            #Readouts can be multiple: max and mean
            self.aig_emb_dim = num_gcn_layer * gnn_hidden_dim * len(readout_type)
            if final_layer_readout == True:
               self.aig_emb_dim = gnn_hidden_dim*len(readout_type)
            self.aig_emb_dim+=2*node_enc_outdim
        else:
            self.aig_emb_dim = 768
        
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.seq_downsample = nn.Linear(512, node_enc_outdim)
        self.glob_stat_upsample = nn.Linear(glob_feature_dim, node_enc_outdim)
        self.denseLayer = nn.Linear(self.aig_emb_dim, n_hidden)
        self.actor = policyNetwork()
        self.critic = valueNetwork()
        torch.nn.init.kaiming_uniform_(self.denseLayer.weight.data)
        torch.nn.init.kaiming_uniform_(self.seq_downsample.weight.data)
        torch.nn.init.kaiming_uniform_(self.glob_stat_upsample.weight.data)
        

    def forward(self, state_data_dict, state=None, info=None):
        graph_data = state_data_dict['graphData']
        glob_feat = state_data_dict['global_feat']
        opt_flow_embedding = state_data_dict['seq_embedding']
        device = next(self.parameters()).device
        glob_feat = numpyToTorch(glob_feat, device)
        opt_flow_embedding = numpyToTorch(opt_flow_embedding, device)
        if self.init_graph_data:
            init_aig_embedding = self.aig(graph_data)
            init_global_embedding = self.glob_stat_upsample(glob_feat)
            seqEmbedding = self.seq_downsample(opt_flow_embedding)
            if init_global_embedding.dim() == 1:
                init_global_embedding = torch.unsqueeze(init_global_embedding, 0)
            if seqEmbedding.dim() == 1:
                seqEmbedding = torch.unsqueeze(seqEmbedding, 0)
            finalEmbedding = torch.cat([init_aig_embedding,seqEmbedding,init_global_embedding],dim=1)
        else:
            finalEmbedding = seqEmbedding
        aigFCOutput = F.leaky_relu(self.denseLayer(finalEmbedding))
        logits, policy = self.actor(aigFCOutput)
        value = self.critic(aigFCOutput)

        #return logits, policy, value,finalEmbedding,init_aig_embedding
        # Standard return for high-level api
        return (logits, value), state
    
class RLStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 256, num_gcn_layers=3,
                 norm_type='batch',final_layer_readout=True, glob_feature_dim = 8,
                 pooling_type=None,pooling_ratio=0.8,readout_type=['mean','max']):
        super().__init__(observation_space, features_dim)
        self.node_encoder = NodeEncoder(features_dim//8)
        self.aig_encoder = AIGEncoder(self.node_encoder,features_dim//8, num_layer=num_gcn_layers)
        self.seq_downsample = nn.Linear(512, features_dim//8)
        self.glob_stat_upsample = nn.Linear(glob_feature_dim, features_dim//8)
        self.dense_layer = nn.Linear(features_dim//2, features_dim)
        torch.nn.init.kaiming_uniform_(self.dense_layer.weight.data)
        torch.nn.init.kaiming_uniform_(self.seq_downsample.weight.data)
        torch.nn.init.kaiming_uniform_(self.glob_stat_upsample.weight.data)

    def forward(self, observations: dict):
        graph_data = observations['graphData']
        glob_feat = observations['global_feat']
        opt_flow_embedding = observations['seq_embedding']
        device = next(self.parameters()).device
        glob_feat = numpyToTorch(glob_feat, device)
        opt_flow_embedding = numpyToTorch(opt_flow_embedding, device)

        init_aig_embedding = self.aig_encoder(graph_data)
        init_global_embedding = self.glob_stat_upsample(glob_feat)
        seqEmbedding = self.seq_downsample(opt_flow_embedding)
        if init_global_embedding.dim() == 1:
            init_global_embedding = torch.unsqueeze(init_global_embedding, 0)
        if seqEmbedding.dim() == 1:
            seqEmbedding = torch.unsqueeze(seqEmbedding, 0)
        finalEmbedding = torch.cat([init_aig_embedding, init_global_embedding, seqEmbedding], dim=1)
        FCOutput = F.leaky_relu(self.dense_layer(finalEmbedding))

        return FCOutput
    
if __name__ == '__main__':
    x = torch.randn(5, 7) 
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    minidata = Data(x, edge_index).to('cuda')
    testnet = AIGEncoder(NodeEncoder(32), 32).to('cuda')
    emb = testnet(minidata)