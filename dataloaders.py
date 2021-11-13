import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class TraceParts_Dataloader(Dataset):
    def __init__(self,sample_points=1024, split='train', normalize=True, cache_size=10000,uniform=True,normals=True,mask=False):
        self.sample_points = sample_points
        self.split = split
        self.normalize = normalize
        self.cache_size = cache_size
        self.uniform = uniform
        self.normals = normals
        self.mask = mask
        self.dataset_path = os.path.join("..","Datasets","TRACEPARTS_DATA")
        train_path = os.path.join(self.dataset_path,"train.csv")
        val_path = os.path.join(self.dataset_path,"validation.csv")
        self.train_files = list(pd.read_csv(train_path)["0"])
        self.val_files = list(pd.read_csv(val_path)["0"])
        self.split_dic = {"train":self.train_files,"val":self.val_files}
        self.cache = {"train":{},"val":{}}
    
    
    def pc_normalize(self,pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    

    def farthest_point_sample(self,point,label,npoint):
        """
        Input:
            xyz: pointcloud data, [N, D]
            label : per point label ,[N]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint, D]
        """
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        label = label[centroids.astype(np.int32)]
        return point,label
        
    def __len__(self):
        return len(self.split_dic[self.split])
    
    def __getitem__(self,index):
        if index in self.cache[self.split]:
            return self.cache[self.split][index][0],self.cache[self.split][index][1]
        else:
            file = np.loadtxt(self.split_dic[self.split][index])
            if self.normals:
                pc,label = file[:, 0:-1],file[:,-1]
            else:
                pc,label = file[:, 0:3],file[:,-1]
            
            if self.uniform:
                pc,label = self.farthest_point_sample(pc,label,self.sample_points)
            else:
                pc,label = pc[0:self.sample_points,:],label[0:self.sample_points]

            if self.normalize:
                pc[:,0:3] = self.pc_normalize(pc[:,0:3])

            
            if len(self.cache) < self.cache_size:
                self.cache[self.split][index] = (pc, label)
            
            return pc,label
                
                
                
            
        
        
        