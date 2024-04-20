import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os, sys
import pickle

class Pouring(Dataset):
    def __init__(
            self, 
            root='datasets/pouring_data',
            graph=False,
            **kwargs
        ):
      
        self.traj_data_ = [] 
        self.labels_ = [] 
        self.graph = graph
        
        for file_ in os.listdir(root):
            with open(os.path.join(root, file_), "rb") as f:
                data = pickle.load(f)
                traj = data['traj']
                traj = traj@np.array(
                        [[
                            [1., 0., 0., data['offset'][0]], 
                            [0., 1., 0., data['offset'][1]], 
                            [0., 0., 1., data['offset'][2]], 
                            [0., 0., 0., 1.]]])
                
                self.traj_data_.append(torch.tensor(traj, dtype=torch.float32).unsqueeze(0))
                self.labels_.append(torch.tensor(data['label']).unsqueeze(0)) # Temporary
                    
        self.traj_data_ = torch.cat(self.traj_data_, dim=0)
        self.labels_ = torch.cat(self.labels_, dim=0)
      
        print(f'Pouring dataset is ready; # of trajectories: {len(self.traj_data_)}')
        
        if self.graph: 
            self.set_graph()
            
    def __getitem__(self, idx):
        if self.graph:
            bs_nn = self.graph['bs_nn']
            if self.graph['include_center']:
                x_c = self.traj_data_[idx]
                x_nn = self.traj_data_[
                    self.dist_mat_indices[
                        idx, 
                        np.random.choice(range(self.graph['num_nn']), bs_nn-1, replace=self.graph['replace'])
                    ]
                ]
                return x_c, torch.cat([x_c.unsqueeze(0), x_nn], dim=0), self.labels_[idx]
            else:
                x_c = self.traj_data_[idx]
                x = self.traj_data_[
                    self.dist_mat_indices[
                        idx, 
                        np.random.choice(range(self.graph['num_nn']), bs_nn, replace=self.graph['replace'])
                    ]
                ]
                return x_c, x, self.labels_[idx]
        else:
            traj = self.traj_data_[idx] # (-, Len, 4, 4)
            labels = self.labels_[idx] # (-, 2) : (pouring style, direction)
            return traj, labels

    def __len__(self) -> int:
        return len(self.traj_data_)
  
    def set_graph(self):
        ############ knn graph construction ############
        # data_temp = self.traj_data_
        # data_temp[:, :, :3, 3] = 0.1*data_temp[:, :, :3, 3]
        # data_temp = data_temp.view(len(self.traj_data_), -1).clone()
        # dist_mat = torch.cdist(data_temp, data_temp)
        # dist_mat_indices = torch.topk(dist_mat, k=self.graph['num_nn'] + 1, dim=1, largest=False, sorted=True)
        # self.dist_mat_indices = dist_mat_indices.indices[:, 1:]
        ################################################
        
        # manually designed graph
        self.dist_mat_indices = torch.tensor(
            [
                [2,4,1,6,3,8,5,7,9],
                [3,5,0,7,2,9,4,6,8],
                [0,4,6,3,1,5,8,7,9],
                [1,5,7,2,0,4,9,6,8],
                [2,6,0,8,5,3,7,1,9],
                [3,7,1,9,4,2,6,0,8],
                [4,8,2,7,5,9,0,3,1],
                [5,9,3,6,4,8,1,2,0],
                [6,4,9,2,7,0,5,3,1],
                [7,5,8,3,6,1,4,2,0]
            ]
        ) 
        
class PouringText(Pouring):
    def __init__(
            self, 
            root='datasets/pouring_data',
            max_text_length=32,
            flatten_texts=True,
            **kwargs
        ):
        super(PouringText, self).__init__(root=root, graph=False, **kwargs)

        self.max_text_length = max_text_length
        
        self.traj_data_temp = [] # traj data (len x 4 x 4)
        self.texts_temp = [] # nautral language text
        self.labels_temp = [] # binary lables (e.g., wine style or not, a lot of water or small, etc)
        
        for file_ in os.listdir(root):
            with open(os.path.join(root, file_), "rb") as f:
                data = pickle.load(f)
                traj = data['traj']
                text = data['text']
                
                traj = traj@np.array(
                        [[
                            [1., 0., 0., data['offset'][0]], 
                            [0., 1., 0., data['offset'][1]], 
                            [0., 0., 1., data['offset'][2]], 
                            [0., 0., 0., 1.]]])
                
                
                self.traj_data_temp.append(torch.tensor(traj, dtype=torch.float32).unsqueeze(0))
                self.texts_temp.append(text)
                self.labels_temp.append(
                    torch.tensor(data['label']).unsqueeze(0)) # Temporary
                    
        self.traj_data_temp = torch.cat(self.traj_data_temp, dim=0)
        self.labels_temp = torch.cat(self.labels_temp, dim=0)     
        
        ### FLATTEN ALL TEXTS ###
        if flatten_texts:
            self.traj_data, self.labels, self.texts = self.flatten_texts(
                self.traj_data_temp,
                self.labels_temp,
                self.texts_temp
            )
            indices = torch.randperm(len(self.traj_data)) 
        else:
            self.traj_data = self.traj_data_temp
            self.labels = self.labels_temp
            self.texts = self.texts_temp
            indices = torch.randperm(len(self.traj_data)) 
        ##########################
        
        self.traj_data_ = self.traj_data[indices]
        self.labels_ = self.labels[indices]
        self.texts_ = []
        for i in indices:
            self.texts_.append(self.texts[i.item()])
        
        print(f'PouringText dataset is ready; # of trajectories: {len(self.traj_data_)}')
        ### SET DICTIONARY FOR EVALUATIONS
        if flatten_texts:
            self.dict_for_evals = self.get_dict_for_evals()
            
    def __getitem__(self, idx):
        traj = self.traj_data_[idx] # (Len, 4, 4)
        labels = self.labels_[idx] # (2) - (pouring style, amount)
        text = self.texts_[idx]
        return traj, text, labels

    def __len__(self) -> int:
        return len(self.traj_data_)
    
    def flatten_texts(self, traj_data_temp, labels_temp, texts_temp):
        traj_data = []
        labels = []
        texts = []
        for tr, la, te in zip(traj_data_temp, labels_temp, texts_temp):
            number_of_texts = len(te)
            traj_data.append(tr.unsqueeze(0).repeat(number_of_texts, 1, 1, 1))
            labels.append(la.unsqueeze(0).repeat(number_of_texts, 1))
            texts += te
        return torch.cat(traj_data, dim=0), torch.cat(labels, dim=0), texts
        
    def get_dict_for_evals(self):
        dict_for_evals = {}
        for te, tr in zip(self.texts, self.traj_data):
            if te in dict_for_evals.keys():
                dict_for_evals[te] += [tr.unsqueeze(0)]
            else:
                dict_for_evals[te] = [tr.unsqueeze(0)]
                
        for key, item in dict_for_evals.items():
            dict_for_evals[key] = torch.cat(item, dim=0)
        return dict_for_evals