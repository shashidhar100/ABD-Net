import torch
from torch import nn
import torch.nn.functional as F

import layers


class ABDNet(nn.Module):
    def __init__(self,d_model_list=[512,512,512],dff=1024,num_heads=4,input_dims=3,if_ffn=True,
                 if_layer_norm=True,if_dropout = True,dropout_p = 0.5,mlp=[64,256,512],n_neighbors=32,
                no_of_classes = 4,if_first_hd=True,first_hd_dim=64,if_group_norm=True,if_avg_pool=True):
        super(ABDNet, self).__init__()
        self.d_model_list = d_model_list
        self.dff = dff
        self.num_heads = num_heads
        self.input_dims = input_dims
        self.if_ffn = if_ffn
        self.if_layer_norm = if_layer_norm
        self.if_dropout = if_dropout
        self.dropout_p = dropout_p
        self.mlp = mlp
        self.n_neighbors = n_neighbors
        self.no_of_classes  = no_of_classes
        self.if_first_hd = if_first_hd
        self.first_hd_dim = first_hd_dim
        self.if_group_norm = if_group_norm
        self.if_avg_pool = if_avg_pool
        
        
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if self.if_first_hd:
            input_dims = self.first_hd_dim + 3
        else:
            input_dims = self.input_dims 
        last_channel = input_dims 
        for out_channel in self.mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.encoders = nn.ModuleList()
        input_dims = self.mlp[-1]
        for i in range(len(self.d_model_list)):
            self.encoders.append(layers.Encoder(d_model = self.d_model_list[i],
                                         dff = self.dff,
                                         num_heads = self.num_heads,
                                         input_dims = input_dims,
                                         if_ffn = self.if_ffn,
                                        if_layer_norm = self.if_layer_norm,
                                        if_dropout = self.if_dropout,
                                        dropout_p = self.dropout_p
                                        ))
            input_dims = self.d_model_list[i]
        if self.if_first_hd:
            self.first_hd_conv = torch.nn.Conv1d(self.input_dims,self.first_hd_dim,1)
            self.first_hd_bn = nn.BatchNorm1d(self.first_hd_dim)
        self.classifier_layer = torch.nn.Conv1d(self.d_model_list[-1],self.no_of_classes,1)
        
            
            
    def forward(self,pc,mask):
        pc = pc.permute(0, 2, 1) #(B,C+d,N) --> (B,N,C+d)
        B,N,D = pc.size()
        
        xyz = pc[:,:,:3]
        if D>3:
            feat = pc[:,:,3:]
        
        pc = pc.permute(0, 2, 1) #(B,C+d,N) --> (B,N,C+d)
        if self.if_first_hd:
            pc = self.first_hd_conv(pc)
            pc = self.first_hd_bn(pc)
            feat = pc.permute(0, 2, 1)
        
        

        knn_idx = layers.knn(xyz,self.n_neighbors)
        grouped_xyz = layers.index_points(xyz,knn_idx)
        if self.if_first_hd:
            grouped_feat = layers.index_points(feat,knn_idx)  
        
        if self.if_group_norm:
            grouped_xyz_norm = layers.group_norm(xyz,grouped_xyz)
        else:
            grouped_xyz_norm = grouped_xyz

        if self.if_first_hd:
            feat_xyz = torch.cat((grouped_xyz_norm,grouped_feat),dim=3)
        else:
            feat_xyz = grouped_xyz
        
        feat_xyz = feat_xyz.permute(0,3,2,1)
        for i,conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat_xyz = F.relu(bn(conv(feat_xyz)))
        
        if self.if_avg_pool:
            feat_xyz = torch.mean(feat_xyz,dim=2)
            
        else:
            feat_xyz = torch.max(feat_xyz,dim=2)[0]
        feat_xyz = feat_xyz.permute(0,2,1)
        
        for i,encoder in enumerate(self.encoders):
            feat_xyz = encoder(feat_xyz,mask)
        feat_xyz = feat_xyz.permute(0,2,1)
        output = F.log_softmax(self.classifier_layer(feat_xyz),dim=1)
        
        output = output.permute(0,2,1)
        
        return output



class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss