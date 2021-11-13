import argparse
import os
import dataloaders
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
# import plotly.graph_objects as go

import torch.nn as nn
import torch
import torch.nn.functional as F
import models 


class ToTensor(object):
  def __call__(self, pointcloud):
    return torch.from_numpy(pointcloud)

# def visualize_pc(x,y,z):
#   fig = go.Figure(go.Scatter3d(x=x, y=y, z=z,
#                              mode="markers",
#                              marker=dict(size=2, symbol="circle", color="darkblue")))
#   return fig

# def seg_pointcloud(pointcloud, label):
#     x,y,z=pointcloud.T
#     c = label.T
#     print(c)
#     def SetColor(x):
#         if(x == 0):
#             return "black"
#         elif(x == 1):
#             return "blue"
#         elif(x == 2):
#             return "magenta"
#         else:
#             return "green"
#     fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, 
#                                        mode='markers',
#                                        marker=dict(
#             size=30,
#             color=list(map(SetColor, c)),               # set color to an array/list of desired values
#             colorscale='Viridis',   # choose a colorscale
#             opacity=1.0
#         ))])
#     fig.update_traces(marker=dict(size=4,
#                                   line=dict(width=2,
#                                             color='DarkSlateGrey')),
#                       selector=dict(mode='markers'))

#     '''
#     fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[-1,1],),
#                      yaxis = dict(nticks=1, range=[-1,1],),
#                      zaxis = dict(nticks=1, range=[-1,1],),),
#     width=700,
#     #margin=dict(r=20, l=10, b=10, t=10)
#     )
#     '''
#     fig.show()


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


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--normals', type=bool, default=False, help='If normals neeeded [default: False]')


    #parser.add_argument('--npoint', type=int,  default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--npoint', type=int,  default=1024, help='Point Number [changed to:8096]')

    return parser.parse_args()



def main(args):
    if args.normals:
        channel = 6
        decomposer = models.Decomposer_v1(input_dims=channel)
        path = "/home/cvg-ws2/Akshay_Sidd_shashi_internship/3d/Decomposer/Results/decomposer_experiment/decomposer_v1_1024_NT"
        checkpoint = torch.load(path + '/checkpoints/best_model.pth')
        decomposer.load_state_dict(checkpoint['model_state_dict'])
        #print(segmenter)
        decomposer.eval()
    else:
        channel  = 3
        path = "/home/cvg-ws2/Akshay_Sidd_shashi_internship/3d/Decomposer/Results/decomposer_experiment/decomposer_v1_1024_NF"
        decomposer = models.Decomposer_v1(input_dims=channel)
        checkpoint = torch.load(path + '/checkpoints/best_model.pth')
        decomposer.load_state_dict(checkpoint['model_state_dict'])
        #print(segmenter)
        decomposer.eval()

    summary_file = open(path + "/different_points_eval.txt","w")


    n_points_list = [1024,2048,4096,8096]
    for i in range(len(n_points_list)):
        print("Points ",n_points_list[i])
        TEST_DATASET = dataloaders.TraceParts_Dataloader(split='val',sample_points=n_points_list[i], normalize=True, cache_size=150,uniform=True,normals=args.normals)
        test_data = torch.utils.data.DataLoader(TEST_DATASET,batch_size=args.batch_size,shuffle=False, num_workers=2,prefetch_factor=4)
        with torch.no_grad():
            m_correct=[]
            for (batch,(points,target)) in tqdm(enumerate(test_data),total=len(test_data),smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, target = points.float(), target.long()
                points = points.transpose(2, 1)
                decomposer = decomposer.eval()
                #seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                seg_pred = decomposer(points,mask=None)
                
                #xxx = seg_pred.contiguous().view(-1, num_part)
                seg_pred = seg_pred.contiguous().view(-1, 4)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).sum()
                m_correct.append(correct.item() / (args.batch_size * args.npoint))
                # test_loss = criterion(seg_pred,target)

            test_instance_acc = np.mean(m_correct)
            print('Validation accuracy is: %.5f' % test_instance_acc)
            summary_file.write("Points {} validation accuracy: {}".format(n_points_list[i],test_instance_acc))
    summary_file.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
# #exit()
# #pointcloud_normal = np.array(pointcloud_normal, dtype="float32")
# pc = ToTensor()(pc)
# with torch.no_grad():
#     seg_preds, feats = segmenter(pc.view(1,pc.shape[0],6).float().transpose(1,2).cuda())
# print("prediction shape:",seg_preds.shape)
# print("feats shape:", feats.shape)
# _, pred = torch.max(seg_preds[0].data,1)

# print("pred labels shape:", pred.shape)
# print(pred)
# print("unique:",np.unique(pred.cpu()))


# #tsne plot
# from sklearn.manifold import TSNE
# import plotly.express as px
# import pandas as pd

# # y = np.asarray(label, dtype='S')
# feats = feats[0].cpu().numpy()

# print(feats.T.shape)
# tsne_2d = TSNE(2, n_jobs=-2).fit_transform(feats.T)
# tsne_2d_dic = {"dim0":tsne_2d[:, 0], "dim1":tsne_2d[:, 1]}

# tsne_2d_df = pd.DataFrame.from_dict(tsne_2d_dic)
# fig = px.scatter(tsne_2d_df, x='dim0', y='dim1')
# fig.write_html("./tsne_plot/unc.html")


# '''
# acc2 = (np.array(pred.cpu())==label)
# print(acc2)
# resulting_acc2 = (np.sum(acc2) / 8096 )*100
# print("accuracy:", resulting_acc2)
# '''

# #seg_pointcloud(pointcloud, label)
# # seg_pointcloud(pointcloud, pred)

# name = "./tsne_plot/unc"
# save_as_ply(name, pc, pred)




