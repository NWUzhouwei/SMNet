# Parametric Networks for 3D Point Cloud Classification
from pointnet2_ops import pointnet2_utils

try:
    from .model_utils import *
except:
    from model_utils import *

import torch.nn.init as init
import torch.nn.functional as F

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x


class MAA(nn.Module):

    def __init__(self, in_channels, group_num, features_num):
        super(MAA, self).__init__()
        self.in_channels = in_channels
        self.group_num = group_num
        self.features_num = features_num
        self.alpha_list = []
        self.beta_list = []
        for i in range(self.features_num):
            self.alpha_list.append(nn.Parameter(torch.ones([1, self.in_channels, 1])))
            self.beta_list.append(nn.Parameter(torch.zeros([1, self.in_channels, 1])))
        self.alpha_list = nn.ParameterList(self.alpha_list)
        self.beta_list = nn.ParameterList(self.beta_list)

        self.linear = Linear1Layer(self.in_channels, self.in_channels, bias=False)

    def forward(self, features_list):
        assert len(features_list) == self.features_num
        for i in range(self.features_num):
            # mean = torch.mean(features_list[i], dim=1, keepdim=True)
            # std = torch.std(features_list[i] - mean)
            # features_list[i] = (features_list[i] - mean) / (std + 1e-5)
            #This code was written by Wang Qian, please pay attention to copyright
            features_list[i] = self.alpha_list[i] * features_list[i] + self.beta_list[i]

        features_list = torch.stack(features_list).sum(dim=0)
        features_list = self.linear(features_list)
        return features_list


# Local Geometry Aggregation
class MLGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, block_num, dim_expansion, surface_points, group_num):
        super().__init__()
        self.surface_points = surface_points
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)
        if dim_expansion == 1:
            expand = 2
        elif dim_expansion == 2:
            expand = 1
        self.linear1 = Linear1Layer(out_dim * expand, out_dim, bias=False)
        self.linear2 = []
        for i in range(block_num):
            self.linear2.append(Linear2Layer(out_dim, bias=True))
        self.linear2 = nn.Sequential(*self.linear2)

        self.Pooling = Pooling()

        self.norm_embedding = Linear1Layer(3, out_dim, bias=False)
        self.curv_embedding = Linear1Layer(3, out_dim, bias=False)
        self.MAA = MAA(out_dim, group_num, 3)
        self.Transform_Normal=Transform_Net()
        self.Transform_Cur=Transform_Net()
    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        # Surface Normal and Curvature
        if self.surface_points is not None:
            est_normal, est_curvature = get_local_geo(knn_xyz[..., :self.surface_points, :])
            est_normal=self.Transform_Normal(est_normal.permute(0,2,1))
            est_curvature=self.Transform_Cur(est_curvature.permute(0,2,1))
            # print(est_normal.shape)
            # print(est_curvature.shape)
            est_normal = self.norm_embedding(est_normal.permute(0,2,1))
            est_curvature = self.curv_embedding(est_curvature.permute(0,2,1))

        # Normalization
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Linear
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x = self.linear1(knn_x.reshape(B, -1, G * K)).reshape(B, -1, G, K)

        # Geometry Extraction
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        # Linear
        for layer in self.linear2:
            knn_x_w = layer(knn_x_w)

        # Pooling
        knn_x_w = self.Pooling(knn_x_w)

        if self.surface_points is not None:
            knn_x_w = self.MAA([knn_x_w, est_normal, est_curvature])
        return knn_x_w


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        return lc_x


# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


# Linear Layer 2
class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels / 2),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm2d(int(in_channels / 2)),
            self.act
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channels / 2), out_channels=in_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        
        # self.k = 3

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)
        x_yuan=x
        # print(x.shape)
        x = self.conv1(x)                       # (batch_size, 3, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        # x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)
        x=torch.bmm(x_yuan.permute(0,2,1), x)+x_yuan.permute(0,2,1)
        return x

def get_local_geo(knn_xyz):
    # Surface Normal and Curvature
    centroid = knn_xyz.mean(dim=2, keepdim=True)
    matrix1 = torch.matmul(centroid.permute(0, 1, 3, 2), centroid)
    matrix2 = torch.matmul(knn_xyz.permute(0, 1, 3, 2), knn_xyz) / knn_xyz.shape[2]
    matrix = matrix1 - matrix2
    u, s, v = torch.linalg.svd(matrix)
    est_normal = v[:, :, :, 2]
    est_normal = est_normal / torch.norm(est_normal, p=2, dim=-1, keepdim=True)
    est_curvature = s + 1e-9
    est_curvature = est_curvature / est_curvature.sum(dim=-1, keepdim=True)
    return est_normal, est_curvature


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.cat([sin_embed, cos_embed], -1)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w


# Parametric Encoder
class MGE(nn.Module):
    def __init__(self, in_channels, input_points, num_stages, embed_dim, k_neighbors,
                 k_neighbors_list, alpha, beta, MLGA_block, dim_expansion):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta

        # Raw-point Embedding
        self.raw_point_embed = Linear1Layer(in_channels, self.embed_dim, bias=False)

        self.FPS_kNN_list = nn.ModuleList()  # FPS, kNN
        self.MLGA_list = nn.ModuleList()  # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList()  # Pooling

        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * dim_expansion[i]
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors[i]))
            self.MLGA_list.append(MLGA(out_dim, self.alpha, self.beta, MLGA_block[i], dim_expansion[i],
                                       surface_points=k_neighbors_list[i], group_num=group_num))

    def forward(self, xyz, x):

        # Raw-point Embedding
        # pdb.set_trace()
        x = self.raw_point_embed(x)

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            x = self.MLGA_list[i](xyz, lc_x, knn_xyz, knn_x)

        # Global Pooling
        x = x.max(-1)[0] + x.mean(-1)
        return x


# Parametric Network for ScanObjectNN
class SMNet(nn.Module):
    def __init__(self, in_channels=4, class_num=15, input_points=1024, num_stages=4,
                 embed_dim=90, k_neighbors=[40,20,20,10], k_neighbors_list=[40, 20, 10, None], beta=100, alpha=1000,
                 MLGA_block=[2, 1, 1, 1], dim_expansion=[2, 2, 2, 1]):
        super().__init__()
        # Parametric Encoder
        self.MGE = MGE(in_channels, input_points, num_stages, embed_dim, k_neighbors, k_neighbors_list,
                       alpha, beta, MLGA_block, dim_expansion)
        self.out_channel = embed_dim
        for i in dim_expansion:
            self.out_channel *= i
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, class_num)
        )

    def forward(self, x, xyz):
        # xyz: point coordinates
        # x: point features

        # Parametric Encoder
        x = self.MGE(xyz, x)

        # Classifier
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    pass
