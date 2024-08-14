import torch
import torch.nn as nn
import torch.nn.functional as F 
import dgl
import dgl.function as fn
from dgl.nn import HGTConv
import numpy as np
import json
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import dgl.nn.pytorch as dglnn
#import argparse

torch.manual_seed(10)
np.random.seed(10)

"""
# Argparse 설정 
parser = argparse.ArgumentParser(description='Train a GCN model')
parser.add_argument('--batch', type=int, default=64, help='Batch size')
parser.add_argument('--dims', type =int, default=128, help= 'Embedding dimensions')
parser.add_argument('--hiddem',type=int, default=128, help= 'Hidden layer size')
parser.add_argument('--lr', type=float, default=0.0001, help= 'Learning rate')
parser.add_argument('--epochs', type=int, default=40, help= 'Number of epochs')
parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')
args = parser.parse_args()
"""
class Args:
    def __init__(self):
        self.batch = 64
        self.dims = 128
        self.hidden = 128
        self.lr = 0.0001
        self.epochs = 1000
        self.early_stop = 5
args = Args()
# 데이터셋 로딩 
class ReviewDataset(Dataset):
    def __init__(self,file_path):
        with open(file_path,'r') as file:
            data = json.load(file)
        
        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[1], sample[0]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def load_data(train_path, val_path, test_path):
    train_data = ReviewDataset(train_path)
    val_data = ReviewDataset(val_path)
    test_data = ReviewDataset(test_path)
    return train_data, val_data, test_data

# Custom collate function to handle variable-length lists
def collate_fn(batch):
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    ratings = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    items = torch.tensor([x[2] for x in batch], dtype=torch.long)
    
    return user_ids, items, ratings

# 그래프 생성 함수
def build_graph(data, user_embeddings, item_embeddings, rating_filter=None):
    user_item_edges = []
    item_user_edges = []
    user_nodes = set()
    item_nodes = set()
    
    for user_idx, item_idx, rating in data:
        user_nodes.add(user_idx)
        item_nodes.add(item_idx)
        if rating_filter is None or rating in rating_filter:
            user_item_edges.append((user_idx,item_idx))
            item_user_edges.append((item_idx,user_idx))
           
    
    # 중복 제거
    user_nodes = sorted(user_nodes)
    item_nodes = sorted(item_nodes)
  
    num_nodes_dict = {'user': len(user_embeddings), 'item': len(item_embeddings)}
    
    g = dgl.heterograph({
        ('user','interacts','item'): (torch.tensor([e[0] for e in user_item_edges]), torch.tensor([e[1] for e in user_item_edges])),
        ('item','re_interacts','user'): (torch.tensor([e[0] for e in item_user_edges]), torch.tensor([e[1] for e in item_user_edges]))},
        num_nodes_dict = num_nodes_dict
    )
    
    # Print graph node counts
    #print(f"Number of user nodes in graph: {g.number_of_nodes('user')}")
    #print(f"Number of item nodes in graph: {g.number_of_nodes('item')}")
    
    # Verify that all nodes in graph have embeddings
    #missing_user_nodes = [node for node in g.nodes('user') if node not in user_embedding_dict]
    #missing_item_nodes = [node for node in g.nodes('item') if node not in item_embedding_dict]

    
    g.nodes['user'].data['h'] = torch.stack([torch.tensor(user_embeddings[i], dtype=torch.float32) for i in range(len(user_embeddings))])
    g.nodes['item'].data['h'] = torch.stack([torch.tensor(item_embeddings[i], dtype=torch.float32) for i in range(len(item_embeddings))])
    
    return g

class SimpleHeteroConv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_ntypes, num_eytpes, dropout=0.2, use_norm=True):
        super(SimpleHeteroConv, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.use_norm = use_norm

        self.linear_v = dglnn.TypedLinear(in_size, hidden_size, num_ntypes)
        self.linear_a = dglnn.TypedLinear(hidden_size, out_size, num_ntypes)

        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size)
        if in_size != hidden_size:
            self.residual_w = nn.Parameter(torch.Tensor(in_size, hidden_size))
            nn.init.xavier_uniform_(self.residual_w)

    def forward(self, g, x, ntype, etype, *, presorted=False):
        self.presorted = presorted
        with g.local_scope():
            #print('initial embed: ', x)
            v = self.linear_v(x,ntype)
            v_dict = {'item': v[:len(g.srcdata['h']['item'])], 'user': v[len(g.srcdata['h']['item']):]}
            g.srcdata['v'] = v_dict
            #print('g.srcdata["v"]: ', g.srcdata['v'])
            #print('g.dstdata["h"]: ', g.dstdata['h'])
            g.update_all(fn.copy_u('v','m'), fn.sum('m','h'))
            #print('update g.dstdata["h"]: ', g.dstdata['h'])
            #print('g.dstdata["h"] shape: ', g.dstdata['h']['item'].shape)
            h = torch.cat([g.dstdata['h']['item'].view(-1,self.hidden_size), g.dstdata['h']['user'].view(-1,self.hidden_size)],dim=0)
            h = self.drop(self.linear_a(h, ntype))

            """
            if x.shape != h.shape:
                h = h + (x @ self.residual_w)
            else:
                h = h + x
            """
            if self.use_norm:
                h_dict = {'item': v[:len(g.srcdata['h']['item'])], 'user': v[len(g.srcdata['h']['item']):]}
                h_item = self.norm(h_dict['item'])
                h_user = self.norm(h_dict['user'])
                h = torch.cat([h_item,h_user],dim=0)
                
            return h

class HeteroGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_ntypes, num_etypes):
        super(HeteroGCN, self).__init__()
        self.layer1 = SimpleHeteroConv(in_size, hidden_size, out_size, num_ntypes, num_etypes)
        self.layer2 = SimpleHeteroConv(hidden_size, hidden_size, out_size, num_ntypes, num_etypes)

    def forward(self,g):

        # ntype, etype indices 
        ntype_dict = {ntype: i for i, ntype in enumerate(g.ntypes)}
        ntype_indices = torch.zeros(g.num_nodes(), dtype=torch.long)
        for ntype, idx in ntype_dict.items():
            ntype_indices[g.nodes(ntype)] = idx 
        etype_indices = torch.zeros(g.num_edges(), dtype=torch.long)

        # node features 
        h_dict = g.ndata['h']
        h = torch.cat([h_dict[ntype] for ntype in g.ntypes], dim=0)
        
        h = self.layer1(g,h,ntype_indices,etype_indices)  
        h = F.relu(h)
        h = self.layer2(g,h,ntype_indices,etype_indices)
        h = F.relu(h)

        return h

class RatingPredictor(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_ntypes, num_etypes):
        super(RatingPredictor, self).__init__()
        self.gcn = HeteroGCN(in_size, hidden_size, out_size, num_ntypes, num_etypes)
       
    def forward(self, origianl_graph, graph,clamp=False):
        h= self.gcn(graph)

        #graph.nodes['user'].data['h'] = h[len(graph.srcdata['h']['item']):]
        #graph.nodes['item'].data['h'] = h[:len(graph.srcdata['h']['item'])]
        #print(len(h))
        src, dst = origianl_graph.edges(etype=('user', 'interacts', 'item'))
      
        user_embeds = h[len(graph.srcdata['h']['item']):][src]
        item_embeds = h[:len(graph.srcdata['h']['user'])][dst]
        #print(user_embeds)
        #print(item_embeds)
        
        edge_ratings = torch.sum(user_embeds * item_embeds, dim=1) 

        if clamp: 
            edge_ratings = torch.clamp(edge_ratings,1,5)
        #print(len(edge_ratings))
        return h, edge_ratings 

def extract_ratings(data):
    ratings = []
    for user_id, asin_id, rating in data:
        ratings.append(rating)
    return torch.tensor(ratings)
    
def save_model(model,path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    best_val_loss = float('inf')
    patience = args.early_stop 
    
    g_original = build_graph(train_data, u_review, i_review)
    g_positive = build_graph(train_data, u_p_review, i_p_review, rating_filter={4,5})
    g_negative = build_graph(train_data, u_n_review, i_n_review, rating_filter={1,2,3})
   
   
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        
        h, pred_original = model(g_original, g_original)
        #debug_printing(h)
        _, pred_positive = model(g_original, g_positive)
        _, pred_negative = model(g_original, g_negative) 
        
        predictions = pred_original + pred_positive - pred_negative 
        #print("predictions total: ", len(predictions))
        #print('prediction ratings: ', predictions)
        loss = criterion(predictions, train_ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        mae_loss, mse_loss, rmse_loss = evaluate(model, val_data,val_ratings, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val MAE Loss: {mae_loss:.4f}, MSE Loss: {mse_loss:.4f}, RMSE Loss: {rmse_loss:.4f}')
        
        if mse_loss < best_val_loss:
            best_val_loss = mse_loss
            best_model = model.state_dict()
            save_path = f'./saved_model/{dataset}/{graph}/e{epoch}_model'
            patience = args.early_stop
        else:
            patience -= 1
            if patience == 0:
                print('Early stopping triggered')
                break 
        save_model(model,save_path)

def evaluate(model, dataset,ratings, criterion,clamp=False):
    model.eval()

    with torch.no_grad():
        
        g_original = build_graph(dataset, u_review, i_review)
        g_positive = build_graph(dataset, u_p_review, i_p_review, rating_filter={4,5})
        g_negative = build_graph(dataset, u_n_review, i_n_review, rating_filter={1,2,3})
        
        
        _, pred_original = model(g_original, g_original,clamp)
        _, pred_positive = model(g_original, g_positive,clamp)
        _, pred_negative = model(g_original, g_negative,clamp) 

        predictions = pred_original + pred_positive - pred_negative 

        mae_loss = (ratings - predictions).abs().mean()
        mse_loss = criterion(predictions, ratings)
        rmse_loss = np.sqrt(mse_loss.item())

    return mae_loss, mse_loss.item(), rmse_loss

# 데이터 로드 경로 
dataset = 'SO'
graph = 'user_item_graph'
train_path = f'./{dataset}/train.json'
val_path = f'./{dataset}/val.json'
test_path = f'./{dataset}/test.json'

train_data, val_data, test_data = load_data(train_path, val_path, test_path)
#print(len(train_data.data))

# 총 사용자와 아이템 수 
num_users = max([int(user_id) for user_id, _, _ in train_data.data]) + 1
num_items = max([asin for _, asin, _ in train_data.data]) + 1

# ratings
train_ratings = extract_ratings(train_data)
val_ratings = extract_ratings(val_data)
test_ratings = extract_ratings(test_data)

#train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
#val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
#test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 리뷰 임베딩 로드 
u_review = torch.tensor(np.load(f'./{dataset}/encoded_user_review_v2.npy', allow_pickle=True))
u_p_review =  torch.tensor(np.load(f'./{dataset}/encoded_user_review_rating_4_or_above_v2.npy', allow_pickle=True))
u_n_review =  torch.tensor(np.load(f'./{dataset}/encoded_user_review_rating_3_or_below_v2.npy', allow_pickle=True))
i_review =  torch.tensor(np.load(f'./{dataset}/encoded_item_review_v2.npy', allow_pickle=True))
i_p_review =  torch.tensor(np.load(f'./{dataset}/encoded_item_review_rating_4_or_above_v2.npy', allow_pickle=True))
i_n_review =  torch.tensor(np.load(f'./{dataset}/encoded_item_review_rating_3_or_below_v2.npy', allow_pickle=True))

#print('i_review: ', i_review[0]) 

# train the model 
model = RatingPredictor(in_size = 768, hidden_size = args.hidden, out_size =args.dims, num_ntypes=2, num_etypes=1)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

print("start training....\n")
print(f'dataset={dataset},graph={graph}\n')
train(model, train_data, val_data, optimizer, criterion, args.epochs)
test_mae, test_mse, test_rmse = evaluate(model, test_data, test_ratings, criterion,clamp=True) 
print(f'Test MAE Loss: {test_mae:.4f}, Test MSE Loss: {test_mse:.4f},Test RMSE Loss: {test_rmse:.4f}')