import argparse
import os
#from data_utils.PrimitiveDataLoader import PartNormalDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
#import provider
import numpy as np
import plotly.graph_objects as go

import torch.nn as nn
import torch
import torch.nn.functional as F

import models



class ToTensor(object):
  def __call__(self, pointcloud):
    return torch.from_numpy(pointcloud)

def visualize_pc(x,y,z):
  fig = go.Figure(go.Scatter3d(x=x, y=y, z=z,
                             mode="markers",
                             marker=dict(size=2, symbol="circle", color="darkblue")))
  return fig

def seg_pointcloud(pointcloud, label):
    x,y,z=pointcloud.T
    c = label.T
    print(c)
    def SetColor(x):
        if(x == 0):
            return "black"
        elif(x == 1):
            return "blue"
        elif(x == 2):
            return "magenta"
        else:
            return "green"
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       marker=dict(
            size=30,
            color=list(map(SetColor, c)),               # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=1.0
        ))])
    fig.update_traces(marker=dict(size=4,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    '''
    fig.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=[-1,1],),
                     yaxis = dict(nticks=1, range=[-1,1],),
                     zaxis = dict(nticks=1, range=[-1,1],),),
    width=700,
    #margin=dict(r=20, l=10, b=10, t=10)
    )
    '''
    fig.show()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    #B, N, C = batch_data.shape
    N, C = batch_data.shape
    assert(clip > 0)
    #jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def save_as_ply(name, pc, pred):
    pred = np.asarray(pred.cpu())
    pc = np.asarray(pc.cpu())
    print(pred)
    pred = pred.reshape(pred.shape[0],1)
    print(pred.shape)
    file = open(name+".ply",'w')
    n_verts = pc.shape[0]
    file.write("ply\nformat ascii 1.0\nelement vertex "+str(n_verts)+"\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    color_map = {0:"0 0 0", 1:"0 0 255", 2:"255 0 255", 3:"0 255 0"}
    encode_pred = ''
    for i,label in enumerate(pred):
        ek = str(pc[i][0])
        do = str(pc[i][1])
        ten = str(pc[i][2])
        char = str(pc[i][3])
        pan = str(pc[i][4])
        she = str(pc[i][5])
        rgb = color_map[label[0]]
        file.write(ek+" "+do+" "+ten+" "+char+" "+pan+" "+she+" "+rgb+'\n')

    file.close()




def save_as_ply_for_attention(name, pc, pred, target_point):
    #pred = np.asarray(pred.cpu())
    #pc = np.asarray(pc.cpu())
    #print(pred)
    pred = pred.reshape(pred.shape[0],1)
    #print(pred.shape)
    file = open(name+".ply",'w')
    n_verts = pc.shape[0]
    file.write("ply\nformat ascii 1.0\nelement vertex "+str(n_verts)+"\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    color_map = {0:"0 0 0", 1:"0 0 255", 2:"255 0 255", 3:"0 255 0"}
    file.write(str(target_point[0])+" "+str(target_point[1])+" "+str(target_point[2])+" "+str(target_point[3])+" "+str(target_point[4])+" "+str(target_point[5])+" "+"255 0 0\n")
    for i,label in enumerate(pred):
        ek = str(pc[i][0])
        do = str(pc[i][1])
        ten = str(pc[i][2])
        char = str(pc[i][3])
        pan = str(pc[i][4])
        she = str(pc[i][5])
        rgb = color_map[label[0]]
        file.write(ek+" "+do+" "+ten+" "+char+" "+pan+" "+she+" "+rgb+'\n')

    file.close()


def rotate_pc_with_normal(batch_xyz_normal):

    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    shape_pc = batch_xyz_normal[:,0:3]
    shape_normal = batch_xyz_normal[:,3:6]
    batch_xyz_normal[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    batch_xyz_normal[:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)#.to(device)
    distance = torch.ones(B, N)* 1e10#.to(device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long)#.to(device)
    batch_indices = torch.arange(B, dtype=torch.long)#.to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    print("fps")
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
    return point


if True:
    channel = 6
    decomposer = models.Decomposer_v1(input_dims=channel,n_neighbors=32)
    path = os.path.join("..","Results","decomposer_experiment","decomposer_v1_1024_NT")
    checkpoint = torch.load(os.path.join(path,"checkpoints","best_model.pth"), map_location=torch.device('cpu'))
    decomposer.load_state_dict(checkpoint['model_state_dict'])
    #decomposer.cuda()
    decomposer = decomposer.eval()
# else:
#     channel  = 3
#     path = os.path.join("..","Results","decomposer_experiment","decomposer_v1_1024_NF")
#     decomposer = models.Decomposer_v1(input_dims=channel,n_neighbors=128)
#     checkpoint =  torch.load(os.path.join(path,"checkpoints","best_model.pth"))
#     decomposer.load_state_dict(checkpoint['model_state_dict'])
#     decomposer.cuda()
#     decomposer = decomposer.eval()



'''
root = 'data/SHAPE_DATASET_NORMALS/'
num = 160
test_data = PartNormalDataset(root = root, npoints=2048, split='val', normal_channel=True)
pc, cat, label = test_data[num]
pointcloud = pc[:,0:3]
print(cat)
'''

############################


'''
#our primitive shapes
#pset = np.loadtxt('./data/SHAPE_DATASET_NORMALS/plane/test_plane_std0_r20.txt')
#pset = np.loadtxt('./data/SHAPE_DATASET_NORMALS/sphere/test_sphere_std0_r20.txt')
#pset = np.loadtxt('./data/SHAPE_DATASET_NORMALS/cone/test_cone_std2_r13.txt')
pset = np.loadtxt('./data/SHAPE_DATASET_NORMALS/cylinder/test_cylinder_std2_r10.txt')
pset[:, 0:3] = pc_normalize(pset[:, 0:3])
pc = pset[:,:6]
pointcloud = pset[:,0:3]
label = pset[:,-1]
'''

############################


'''
#our primitive shapes combined
#t = np.loadtxt('./data/test_pc.txt')
t = np.loadtxt('./data/test_pc2.txt')
print(t.shape)
####choice = np.random.choice(4096, 2048, replace=False)
####pset = t[choice, :]
####seg = t[:, -1]
####label = seg[choice]
pset = t
pset[:, 0:3] = pc_normalize(pset[:, 0:3])
pc = pset[:,:6]
pointcloud = pset[:,0:3]
'''

############################

'''
#from modelnet40
#airplane
#t = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data\\modelnet40_normal_resampled/airplane/airplane_0003.txt', delimiter=',')

# t = np.loadtxt('./data/modelnet40_normal_resampled/piano/piano_0005.txt', delimiter=',')


# t = np.loadtxt('./data/modelnet40_normal_resampled/airplane/airplane_0027.txt', delimiter=',')
#bathtub
#t = np.loadtxt('./data/modelnet40_normal_resampled/bathtub/bathtub_0014.txt', delimiter=',')

#bed
#t = np.loadtxt('./data/modelnet40_normal_resampled/bed/bed_0014.txt', delimiter=',')

# dresser
# t = np.loadtxt('./data/modelnet40_normal_resampled/dresser/dresser_0050.txt', delimiter=',')

#cone
# t = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data/modelnet40_normal_resampled/cone/cone_0039.txt', delimiter=',')

# chair
#t = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data\\modelnet40_normal_resampled/chair/chair_0921.txt', delimiter=',')

#stool
# t = np.loadtxt('./data/modelnet40_normal_resampled/cone/cone_0030.txt', delimiter=',')

# guitar
# t = np.loadtxt('./data/modelnet40_normal_resampled/guitar/guitar_0051.txt', delimiter=',')


# vase
#t = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data\\modelnet40_normal_resampled/vase/vase_0034.txt', delimiter=',')


#from shapenet
#airplane:02691156
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1a04e3eab45ca15dd86060f189eb133.txt')
#t[:,:6] = rotate_pc_with_normal(t[:,:6])
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1d269dbde96f067966cf1b4a8fc3914e.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/2af93e42ceca0ff7efe7c6556ea140b4.txt')




#bag:02773838
# t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02773838/68e4ba38895d820df6fec4b901a12701.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02773838/10a885f5971d9d4ce858db1dc3499392.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02773838/2022610a5d1a8455abc49cae1a831a9e.txt')


#cap:02954340
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02954340/6f93656d083e985465bae2cb33eb4baa.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02954340/1eccbbbf1503c888f691355a196da5f.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02954340/a4f94067b0ec9608e762cf5917cef4ef.txt')


#car:02958343
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02958343/1e3f494626a24badf35b4953d8add91f.txt')

#chair:03001627
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03001627/1d6f4020cab4ec1962d6a66a1a314d66.txt')


#earphone:03261776
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03261776/a5a29c03bca0ff22908c261630fec33.txt')


#guitar:03467517
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03467517/3bca4bb40d2b8cf23b3435cb12e628d5.txt')



#knife:03624134
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03624134/3ca548ca4f557a4cda3cc9549ae4a7d3.txt')

#lamp: 03636649
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03636649/1a44dd6ee873d443da13974b3533fb59.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03636649/1e83293107d6c3a92cd2160e449d45ae.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03636649/2d4c4ceacdd41cf1f8c0f5916f81d758.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03636649/2fa7dbc66467235e2102429c788ba90.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03636649/676ec374028a24db76e29c9c43bc7aa.txt')


#laptop:03642806
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03642806/4a715660d89e097664c908bd136dd7c0.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03642806/7df09674bc991904c78df40cf2e9097a.txt')


#bike:03790512
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03790512/4f30742005b7c20e883158c0007ed9ba.txt')

#cup:03797390
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390/4b8b10d03552e0891898dfa8eb8eefff.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390/214dbcace712e49de195a69ef7c885a4.txt')

#pistol:03948459
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03948459/3b4def7c9d82c19b4cdad9a5bf52dd5.txt')




#rocket: 04099429
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04099429/d9b0e3cd3ce4507931c6673b192319d8.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04099429/e98fdff7cfdf15edd8e5da8daec55d43.txt')

#skate:04225987
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04225987/70c8f82bf6a6fef2db8b0e67801ff73a.txt')


# name = './results_prim_seg/table3'
#table:04379243
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04379243/1aaaed47c3a77219f2b931201029bc76.txt')
#t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04379243/1b7dd5d16aa6fdc1f716cef24b13c00.txt')
# t = np.loadtxt('./data/shapenetcore_partanno_segmentation_benchmark_v0_normal/04379243/1c259cf6c1206aadb6f6d1d58b7b1311.txt')

'''
#if(t.shape[0]>2048):
#    choice = np.random.choice(t.shape[0], 1024, replace=False)
#    pset = t[choice, :]
#    seg = t[:, -1]
#    label = seg[choice]
'''

t = farthest_point_sample(t, 8096)
pset = t

pset[:, 0:3] = pc_normalize(pset[:, 0:3])
pc = pset[:,:6]
label = pset[:,-1]
pointcloud = pset[:,0:3]
'''



############################


#trace parts dataset

#pset = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data/TRACEPARTS_DATA/primitive/114csnb_bolt_3_4_10_2_868_1_75_51.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/t1crphms_unc_1_4_20_1_375_1.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/sshcs_unc_7_16_14_9_1_375.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/gasket_362_x_320_8_class_1500_iso_pn_250.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/grooved_pin_crf_5_64_1.txt')

abs_p = "F:\\REU\\SA-Decomp_code\\decomp_for_drive\\Decomposer\\Dataset"


#val
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/t3srfhms_unf_no__4_48_0_4375_0_4375.txt')
#pset = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data/TRACEPARTS_DATA/primitive/nut_m5x0_8ansib18_2_4_5m.txt')
#pset = np.loadtxt(abs_p+'data/TRACEPARTS_DATA/primitive/sssscup_unc_6_no__5_40_0_25.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/bushingl_L-32-16.txt')
#pset = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data/TRACEPARTS_DATA/primitive/straight_tee_sch_160_dn_1_2.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/m20x2_5x35ansib18_2_3_1m.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/reducer_inox_sch_40s_dn_3_4_x_1_2.txt')
#pset = np.loadtxt('D:\\REU_Sameer_Siddharth\\pointNet++\\Pointnet_Pointnet2_pytorch-master\\data/TRACEPARTS_DATA/primitive/spring_pin_cold_1_16_13_16.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/heavy_hex_bolt_unf_series_5_8x18x1_5_ansi_b18_2_1.txt')
#pset = np.loadtxt('./data/TRACEPARTS_DATA/primitive/class_2500_flange_nps_3_4_wn_stf_11_06.txt')
#pset = np.loadtxt(abs_p+'data/TRACEPARTS_DATA/primitive/m20x2_5x70ansib18_2_3_1m.txt')
pset = np.loadtxt(abs_p+'/TRACEPARTS_DATA/primitive/t3srfhms_unf_no__4_48_0_4375_0_4375.txt')
#shape_data/primitive/m20x2_5x70ansib18_2_3_1m



pset = farthest_point_sample(pset, 1024)
pset[:, 0:3] = pc_normalize(pset[:, 0:3])
pc = pset[:,:6]
pointcloud = pset[:,0:3]
label = pset[:,-1]


############################

'''
#heritage pointcloud

t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/chunk1.txt')
#t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/chunk2.txt')
#t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/chunk3.txt')
#t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/chunk4.txt')
#t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/chunk5.txt')
# t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/forpillar_mesh1_bb.txt')

# t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/part1.txt')
#t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/part2.txt')
# t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/part3.txt')
# t = np.loadtxt('D:\\REU_Sameer_Siddharth/chunks/part4.txt')

##COMMENTED
shapet = t.shape
if(t.shape[0]):
    choice = np.random.choice(shapet[0], 100000, replace=False)
    pset = t[choice, :]
    seg = t[:, -1]
    label = seg[choice]
else:
    pset = t


pset = t

pset[:, 0:3] = pc_normalize(pset[:, 0:3])

pc = pset[:,:6]
pointcloud = pset[:,0:3]
#label = pset[:,-1]
'''

############################


print("pointcloud with features:", pc.shape)
print("pointcloud shape:", pointcloud.shape)
#print("labels:", label.shape)
#print("labels unique:", np.unique(label))
# visualize_pc(*pointcloud.T).show()



pc = ToTensor()(pc)
with torch.no_grad():
    #seg_preds = decomposer(pc.view(1,pc.shape[0],6).float().transpose(1,2).cuda(), mask=None)
    seg_preds, att_weight_list = decomposer(pc.view(1,pc.shape[0],6).float().transpose(1,2), mask=None)

print("len of weight list:", len(att_weight_list))
print("attention weight shape:", att_weight_list[0].shape)




encoder_num = 1
head_num = 3
attenion = att_weight_list[encoder_num][0][head_num].numpy()
print("selected attention shape:", attenion.shape)
print("unique values:", np.unique(attenion))
print("len of unique values:", len(np.unique(attenion)))



print("prediction shape:",seg_preds.shape)
#print("feats shape:", feats.shape)
_, pred = torch.max(seg_preds[0].data,1)


save_as_ply("att_weight_visualize/c_original", pc, pred)

pred = pred.numpy()
pc = pc.numpy()



top_points_to_select = 100
for idx, pt in enumerate(pc):
    selected_point_index = attenion[idx][:].argsort()[-top_points_to_select:][::-1]
    selected = pc[selected_point_index]

    print(selected.shape)
    save_as_ply_for_attention("att_weight_visualize/c", selected, pred[selected_point_index], pt)
    exit()







print("pred labels shape:", pred.shape)
print("unique:",np.unique(pred.cpu()))

print("")

# print("confusion_matrix:\n", confusion_matrix(label, pred.cpu()))
print("confusion_matrix:\n", confusion_matrix(label, pred))


#seg_pointcloud(pointcloud, label)
# seg_pointcloud(pointcloud, pred)

# name = "./att_weight_visualize/trs_1024_k_32"
name = "./trs_1024_k_32"
save_as_ply(name, pc, pred)




