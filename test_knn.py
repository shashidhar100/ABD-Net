
import torch
import numpy as np
import os
# from dataloaders import TraceParts_Dataloader
# train_data = TraceParts_Dataloader(split="val",sample_points=8096, normals=False)
# train_loader = torch.utils.data.DataLoader(train_data,batch_size=16,shuffle=False)
# print(len(train_loader))
cuda = torch.device('cuda:0')
import time
import cudf
from cuml.neighbors import KNeighborsClassifier as cuKNeighbors
from cuml.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors as skKNeighbors
from typing import List

import cupy
import torch
from torch._vmap_internals import vmap

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

def save_as_ply(name,pc,center,idx):
    #pred = np.asarray(pred.cpu())
    # pred = np.ones((pc.shape[0],1))*3
    #pc = np.asarray(pc.cpu())
    # pred = pred.reshape(pred.shape[0],1)
    file = open(name+".ply",'w')
    n_verts = pc.shape[0]
    file.write("ply\nformat ascii 1.0\nelement vertex "+str(n_verts)+"\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    color_map = {0:"0 0 0", 1:"0 0 255", 2:"255 0 255", 3:"0 255 0"}
    # encode_pred = ''
    file.write(str(center[0])+" "+str(center[1])+" "+str(center[2])+" "+"0 0 255\n")
    for i in range(pc.shape[0]):
        ek = str(pc[i][0])
        do = str(pc[i][1])
        ten = str(pc[i][2])
        #char = str(pc[i][3])
        #pan = str(pc[i][4])
        #she = str(pc[i][5])
        if i in idx:
            rgb = color_map[2]
        else:
            rgb = "0 0 0"
        #file.write(ek+" "+do+" "+ten+" "+char+" "+pan+" "+she+" "+rgb+'\n')
        file.write(ek+" "+do+" "+ten+" "+rgb+'\n')

    file.close()


def pc_normalize(pc):
    centroid = np.mean(pc, axis = 0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis = 1)))
    pc = pc/m
    return pc



# def knn(pc):
#     # pc
#     model = NearestNeighbors(n_neighbors=128)
#     # model = skKNeighbors(n_neighbors=128)
#     model.fit(pc[:,0:3])
#     distances,indices = model.kneighbors(pc[:,0:3])
#     return pc

# @torch.jit.script
# def main(pc):
    
        
        
#     futures : List[torch.jit.Future[torch.Tensor]] = []
#     for i in range(16):
#         futures.append(torch.jit.fork(knn,pc[i]))
#     results = []
#     for future in futures:
#         results.append(torch.jit.wait(future))

#     return
        

# for batch,(pc,label) in enumerate(train_loader):
#     print(batch)
#     s_t = time.time()
#     main(pc)
#     print("Time Taken :",time.time()-s_t)



def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



import time


# def conv(fld):
#     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)

# converters={0: conv, 1: conv, 2: conv, 3: conv, 4: conv, 5: conv}





pset = np.loadtxt("/home/cvg-ws2/Akshay_Sidd_shashi_internship/3d/Decomposer/codes/cup_0005.txt")
print(pset[0][0])
pset[:,0:3] = pc_normalize(pset[:,0:3])
pc = pset[:,0:3]
label = pset[:,-1]

s_t = time.time()
dis = torch.cdist(pc[:,:,:3],pc[:,:,:3])
neighbors = dis.topk(dim=2,k=32,largest=False)
points_with_neighbors  = index_points(pc,neighbors.indices)
print(points_with_neighbors.shape)
# pc_with_neighbors = pc[:,:,:][neighbors.indices[:,:,:]]
# print(pc_with_neighbors.shape)
# print(neighbors.indices[0][0][:].numpy())
save_as_ply(str(batch)+"32",pc[0,:,:3].numpy(),pc[0][0][:3].numpy(),neighbors.indices[0][0][:].numpy())
print("Time Taken :",time.time()-s_t)



# s_t = time.time()
# dis = torch.cdist(pc[:,:,:3],pc[:,:,:3])
# neighbors = dis.topk(dim=2,k=128,largest=False)
# points_with_neighbors  = index_points(pc,neighbors.indices)
# print(points_with_neighbors.shape)
# # pc_with_neighbors = pc[:,:,:][neighbors.indices[:,:,:]]
# # print(pc_with_neighbors.shape)
# # print(neighbors.indices[0][0][:].numpy())
# save_as_ply(str(batch)+"128",pc[0,:,:3].numpy(),pc[0][0][:3].numpy(),neighbors.indices[0][0][:].numpy())
# print("Time Taken :",time.time()-s_t)







# for batch,(pc,label) in enumerate(train_loader):
#     print(batch)
    
#     s_t = time.time()
#     dis = torch.cdist(pc[:,:,:3],pc[:,:,:3])
#     neighbors = dis.topk(dim=2,k=32,largest=False)
#     points_with_neighbors  = index_points(pc,neighbors.indices)
#     print(points_with_neighbors.shape)
#     # pc_with_neighbors = pc[:,:,:][neighbors.indices[:,:,:]]
#     # print(pc_with_neighbors.shape)
#     # print(neighbors.indices[0][0][:].numpy())
#     save_as_ply(str(batch)+"32",pc[0,:,:3].numpy(),pc[0][0][:3].numpy(),neighbors.indices[0][0][:].numpy())
#     print("Time Taken :",time.time()-s_t)

#     print(batch)
    
#     s_t = time.time()
#     dis = torch.cdist(pc[:,:,:3],pc[:,:,:3])
#     neighbors = dis.topk(dim=2,k=128,largest=False)
#     points_with_neighbors  = index_points(pc,neighbors.indices)
#     print(points_with_neighbors.shape)
#     # pc_with_neighbors = pc[:,:,:][neighbors.indices[:,:,:]]
#     # print(pc_with_neighbors.shape)
#     # print(neighbors.indices[0][0][:].numpy())
#     save_as_ply(str(batch)+"128",pc[0,:,:3].numpy(),pc[0][0][:3].numpy(),neighbors.indices[0][0][:].numpy())
#     print("Time Taken :",time.time()-s_t)
    

    # vmap(knn)(pc.numpy())

    # pc = torch.squeeze(pc,dim=0).numpy()
    # dx = to_dlpack(pc)

# Convert it into a CuPy array.
    # pc = cupy.fromDlpack(dx)

# Convert it back to a PyTorch tensor.
    # pc = from_dlpack(cx.toDlpack())
    # pc = cudf.DataFrame(pc)
    # label = cudf.DataFrame(label.numpy())
#     print(label.shape)
    # label = label.numpy()
    # print(pc.shape)
    # print(pc[0,:])
    # x = pc[0,:]
    # model = cuKNeighbors(n_neighbors=32)
    # model = NearestNeighbors(n_neighbors=128)
    # model = skKNeighbors(n_neighbors=128)
    # model.fit(pc[:,0:3])
    # distances,indices = model.kneighbors(pc[:,0:3])
    # print(pc.shape)
    # print(indices[0].shape)
    # print(indices[0])
    # points = pc[:,0:3][indices[0]]
    # save_as_ply(str(batch),pc,pc[0,0:3],indices[0])
    # print(points.shape)

    # file = open(str(batch)+".ply","w")

    
    # print(indices.shape)
    # print("Time Taken :",time.time()-s_t)
    