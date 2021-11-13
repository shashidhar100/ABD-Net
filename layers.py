import torch
from torch import nn
import torch.nn.functional as F



def scaled_dot_product_attention(q,k,v,mask=None):
    """
    q : query matrix (..,len_q,depth)
    k : key matrix (..,len_k,depth)
    v : value matrix (..,len_v,depth_v) len_v == len_k
    mask : mask to be used before softmax (..,len_q,len_k)
    """
    
    qk = torch.matmul(q,torch.transpose(k,-2,-1)) # (..,len_q,len_k)
    
    # scaling the qk
    dk = torch.tensor(k.shape[-1]).float()
    scaled_qk = qk / torch.sqrt(dk)
    # print(scaled_qk.size(),mask.size())
    if mask is not None:
        scaled_qk += (mask * -1e9)
        
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    
    attention_weights = torch.nn.functional.softmax(scaled_qk,dim = -1) #(..,len_q,len_k)
    
    output = torch.matmul(attention_weights,v) #(len_q,depth_v)
    
    return output,attention_weights


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
    # print(view_shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # print(view_shape)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # print(repeat_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # print(batch_indices[0].shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn(pc,n_neighbors=32):
    """
    Input:
        pc : Point cloud (B,N,d) knn is computed in the d dimension eculidean spcae with the euclidean distance
        n_neighbors : Number of neighbors to find default is 32
    Return:
        indices of the n_neighbors for each point (B,N,n_neighbors)
    """

    dist = torch.cdist(pc,pc)
    neigbhors = dist.topk(k=n_neighbors,dim=2,largest=False)
    return neigbhors.indices




class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,input_dims):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model #output feature dimension
        self.input_dims = input_dims
        
        #d_model is the depth of each head hence it should be divisible by number of heads
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        self.wq = torch.nn.Linear(input_dims,d_model)
        self.wk = torch.nn.Linear(input_dims,d_model)
        self.wv = torch.nn.Linear(input_dims,d_model)
        
        self.projector = torch.nn.Linear(d_model,d_model)
        
    
    def split_heads(self,x,batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len(N), depth)
        """
        x = torch.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return torch.transpose(x,2,1)
    
    def forward(self,v,k,q,mask):
        batch_size = q.size()[0]
        
        q = self.wq(q) #(batch_size,seq_len(N),d_model)
        k = self.wk(k) #(batch_size,seq_len(N),d_model)
        v = self.wv(v) #(batch_size,seq_len(N),d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, len_v, depth)
        
        if mask!=None:
            mask = mask.unsqueeze(1)
            mask = torch.repeat_interleave(mask,4,dim=1)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = torch.transpose(scaled_attention,2,1)  # (batch_size, len_q, num_heads, depth)
        
        concat_attention = torch.reshape(scaled_attention,(batch_size, -1, self.d_model))   # (batch_size,len_q, d_model)

        
        output = self.projector(concat_attention)  # (batch_size,len_q, d_model)

        return output, attention_weights



        

        


class point_wise_feed_forward_network(nn.Module):
    def __init__(self,d_model,dff,input_dims):
        super(point_wise_feed_forward_network, self).__init__()
        self.d_model = d_model # output feature dimension
        self.dff = dff # intermediate feature dimension
        self.input_dims = input_dims 
        
        self.dense1 = torch.nn.Linear(self.input_dims,self.dff) # (batch_size, seq_len(N), dff)
        self.dense2 = torch.nn.Linear(self.dff,self.d_model)  # (batch_size, seq_len(N), d_model)
        
        
    def forward(self,x):
        x = F.relu(self.dense1(x))
        output = self.dense2(x)
        return output




class Encoder(nn.Module):
    def __init__(self,d_model,dff,num_heads,input_dims,if_ffn=False,if_layer_norm=False,
                if_dropout = False,dropout_p = 0.5):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.input_dims = input_dims
        self.if_ffn = if_ffn
        self.if_layer_norm = if_layer_norm
        self.if_dropout = if_dropout
        self.dropout_p = dropout_p

        
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads,input_dims = self.input_dims)
        if self.if_ffn:
            self.ffn = point_wise_feed_forward_network(d_model = self.d_model,dff=self.dff,input_dims = self.d_model)
        if self.if_layer_norm:
            self.layernorm1 = torch.nn.BatchNorm1d(self.d_model)
            self.layernorm2 = torch.nn.BatchNorm1d(self.d_model)
        if self.if_dropout:
            self.dropout1 = torch.nn.Dropout(self.dropout_p)
            self.dropout2 = torch.nn.Dropout(self.dropout_p)
        
    
    def forward(self,x,mask):
        out1, _ = self.mha(x, x, x, mask)
        if self.if_dropout:
            out1 = self.dropout1(out1)
        
        if self.if_layer_norm:
            x = x.permute(0,2,1)
            out1 = out1.permute(0,2,1)
            add = x + out1
            out1 = self.layernorm1(add)
            out1 = out1.permute(0,2,1)
            x = x.permute(0,2,1)

        
            
        
        if self.if_ffn:
            out2 = self.ffn(out1)
            if self.if_dropout:
                out2 = self.dropout2(out2)
            if self.if_layer_norm:
                out2 = out2.permute(0,2,1)
                out1 = out1.permute(0,2,1)
                add = out1 + out2
                out2 = self.layernorm2(add)
                out2 = out2.permute(0,2,1)
        return out2
           


def group_norm(points,group_points):
    """
    Inputs:
        points: original points (B,N,3)
        group_points : grouped points (B,N,S,3)

    Returns:
        group_points_norm : points features relative to centers (B,N,S,3)
    """
    B,S,K,C = group_points.size()
    group_points_norm = group_points - points.view(B,S,1,C)
    return group_points_norm


