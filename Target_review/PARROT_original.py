import os
#import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

torch.manual_seed(10) 
np.random.seed(10)
device = torch.device('cuda:0')

"""
#Argparse 설정 
args = argparse.ArgumentParser(description='args')
args.add_argument('--batch', default=128, type=int)
args.add_argument('--dims', default=768, type=int)
args.add_argument('--lr', default=0.0001, type=float)
args.add_argument('--hidden', default=128, type=int)
args.add_argument('--reg_lr', default=1, type=float)
args.add_argument('--CL', default=0, type=int)
args.add_argument('--aa', default=0, type=int)
args.add_argument('--cl_lr', default=1, type=float)
args.add_argument('--aa_lr', default=1, type=float)
args.add_argument('--early_stop', default=5, type=int)
args.add_argument('--dataset', default='DM', type=str)
args = args.parse_args()
"""

class Args:
    def __init__(self):
        self.batch = 64
        self.dims = 768
        self.hidden = 128
        self.lr = 0.0001
        self.epochs = 1000
        self.early_stop = 5
args = Args()

# 데이터셋 로딩 클래스
class ReviewDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[0], sample[1]))
    
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


class MFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, reviews, ratings):
        super(MFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_items = num_items 
        
        self.ReLU = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.reviews = reviews
        self.ratings = ratings 
        self.batch_norm = nn.BatchNorm1d(hidden_dim)


        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.user_embedding.weight = nn.Parameter(torch.from_numpy(reviews[0]))
        self.item_embedding.weight = nn.Parameter(torch.from_numpy(reviews[1]))

        self.user_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False 

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1) 

        self.resFC = nn.Linear(hidden_dim, hidden_dim)
        self.resFC2 = nn.Linear(hidden_dim, hidden_dim)

        self.uFC = nn.Sequential(
                nn.Linear(num_items, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.iFC = nn.Sequential(
                nn.Linear(num_users, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.ruFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )
        self.riFC = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout()
                )

        self.rr_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)

        self.user_rating = torch.from_numpy(ratings[0].astype('float32')).to(device)
        self.item_rating = torch.from_numpy(ratings[1].astype('float32')).to(device)

    def mark_unique_elements(self, tensor):

        unique_elements, inverse_indices, counts = torch.unique(tensor, return_inverse = True, return_counts = True)

        boolean_tensor = torch.zeros_like(tensor, dtype=torch.bool, device = device)
        first_occurrence = torch.zeros_like(unique_elements, dtype=torch.bool, device=device)
        boolean_tensor[first_occurrence[inverse_indices].logical_not()] = True
        first_occurrence[inverse_indices] = True 

        return boolean_tensor 


    def forward(self, user_ids, item_ids, rating, clip):
        unique_user_ids = self.mark_unique_elements(user_ids)
        unique_item_ids = self.mark_unique_elements(item_ids)

        user_r_embeds = self.uFC(self.user_rating / 5)
        item_r_embeds = self.iFC(self.item_rating / 5)

        user_embeds = self.ruFC(self.user_embedding.weight)
        item_embeds = self.riFC(self.item_embedding.weight)

        user_r_embeds, _ = self.rr_attn(user_r_embeds[user_ids],user_r_embeds,user_embeds)
        item_r_embeds, _ = self.rr_attn_2(item_r_embeds[item_ids],item_r_embeds,item_embeds)

        user_r_embeds = user_r_embeds + user_embeds[user_ids]
        item_r_embeds = item_r_embeds + item_embeds[item_ids]

        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()
        
        dot_product = (user_r_embeds * item_r_embeds).sum(1)
        prediction = dot_product + user_biases + item_biases 

        if clip==1:
            prediction = torch.clamp(prediction,1,5)

        return prediction 
        

def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    for user_ids, item_ids, ratings in dataloader:
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

        optimizer.zero_grad()
        outputs = model(user_ids, item_ids, ratings, clip=0)

        loss = criterion(outputs,ratings)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_rmse(model, dataloader, device, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_mae = 0
    total_mse = 0
    total_rmse = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

            outputs = model(user_ids, item_ids, ratings, clip=1)
            mae = (ratings - outputs).abs().mean()
            mse = nn.MSELoss()(outputs,ratings)
            rmse = np.sqrt(mse.item())
            total_mae += mae.item()
            total_mse += mse.item()
            total_rmse = rmse.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    return total_rmse / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)

# 함수: 모델 저장
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 함수: 모델 불러오기
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

def load_ratings(dataset):
    with open(dataset, 'r') as file:
        data = json.load(file)

    # reviewer_index와 asin_index를 찾기 위해 최대 인덱스 값을 구함
    max_reviewer_index = max(int(reviewer) for reviewer in data.keys())
    max_asin_index = max(max(rating_asin[1] for rating_asin in ratings) for ratings in data.values())

    # 평점 벡터 초기화
    reviewer_ratings = np.zeros((max_reviewer_index+1, max_asin_index+1))
    asin_ratings = np.zeros((max_asin_index+1, max_reviewer_index+1))

    # 평점 벡터 생성
    for reviewer_index, ratings in data.items():
        reviewer_index = int(reviewer_index)
        for rating, asin_index in ratings:
            reviewer_ratings[reviewer_index, asin_index] = rating
            asin_ratings[asin_index, reviewer_index] = rating

    return reviewer_ratings.astype('int'), asin_ratings.astype('int')

def main():
    dataset = 'VG'
    graph = 'parrot'
    train_path = f'./{dataset}/train.json'
    val_path = f'./{dataset}/val.json'
    test_path = f'./{dataset}/test.json'

    train_data, val_data, test_data = load_data(train_path, val_path, test_path)

    # 총 사용자, 아이템 수 
    num_users = max([int(user_id) for user_id, _, _ in train_data.data]) + 1
    num_items = max([asin for _, _, asin in train_data.data]) + 1 
    #print(num_items)

    print('Data Load Start....\n')
    
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle = True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, collate_fn = collate_fn)

    u_review = np.load(f'./{dataset}/encoded_user_review_v2.npy', allow_pickle=True)
    i_review = np.load(f'./{dataset}/encoded_item_review_v2.npy', allow_pickle=True)

    u_ratings, i_ratings = load_ratings(train_path)
    
    print('Data Load finished! \n')
    
    model = MFModel(num_users, num_items, embedding_dim = args.dims, hidden_dim=args.hidden, reviews=[u_review,i_review], ratings=[u_ratings,i_ratings]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    early_stop_epochs = args.early_stop
    updated = 0
    best = 100
    
    for epoch in range(args.epochs):
        print('Start training....\n') 
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch)
        val_rmse, val_mse, val_mae = evaluate_rmse(model, val_loader, device, criterion)
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')
        if best > val_mse:
            best = val_mse
            best_path = f'./saved_model/{dataset}/{graph}/MFModel_{epoch}'
            best_model = model.state_dict()
            updated = 0 

        updated += 1 
        if updated > args.early_stop and epoch > 0:
            print('Early stopping triggered')
            break

    save_model(model,best_path)
    model.load_state_dict(best_model)
    test_rmse, test_mse, test_mae = evaluate_rmse(model, test_loader, device, criterion)
    print(f'Test RMSE: {test_rmse:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')

if __name__ == '__main__':
    main()