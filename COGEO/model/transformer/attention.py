import torch.nn as nn
import torch
import numpy as np
from einops import rearrange

class P2I_CrossAttention(nn.Module):
    """
    func: Point to Image Cross Attention with geometric information
    inputs:
        in_dim: input dim 
        out_dim: out dim
    """

    def __init__(self, in_dim, out_dim):
        super(P2I_CrossAttention, self).__init__()
        self.query_conv = nn.Linear(in_dim, out_dim)
        self.key_conv = nn.Linear(in_dim, out_dim)
        self.value_conv = nn.Linear(in_dim, out_dim)
        self.out_conv = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)
        
        # 添加几何信息相关的层
        self.geo_proj = nn.Linear(out_dim, out_dim)
        self.geo_gate = nn.Linear(out_dim * 2, 1)

    def forward(self, feat0, feat1, geo_embed=None):
        '''
        :param feat0: query (point cloud features)
        :param feat1: key & value (image features)
        :param geo_embed: geometric embedding
        :return: attentional value
        '''
        batch_size, C, width, height = feat0.size()
        _, C1, N = feat1.size()
        assert C == C1
        
        # 处理点云特征
        proj_query = self.query_conv(rearrange(feat0, 'b c w h -> b (w h) c'))
        
        # 如果提供了几何信息，将其融合到查询中
        if geo_embed is not None:
            geo_embed = self.geo_proj(geo_embed)
            gate = torch.sigmoid(self.geo_gate(torch.cat([proj_query, geo_embed], dim=-1)))
            proj_query = proj_query + gate * geo_embed

        # 处理图像特征
        proj_key = self.key_conv(rearrange(feat1, 'b c n -> b n c'))
        proj_value = self.value_conv(rearrange(feat1, 'b c n -> b n c'))

        # 计算注意力
        energy = torch.bmm(proj_query, rearrange(proj_key, 'b n c -> b c n'))
        attention = self.softmax(energy) / C ** .5

        # 应用注意力
        out = self.out_conv(torch.bmm(attention, proj_value))
        out = rearrange(out, 'b (w h) c -> b c w h', w=width, h=height)
        
        # 如果提供了几何信息，调制输出
        if geo_embed is not None:
            geo_embed_reshaped = rearrange(geo_embed, 'b (w h) c -> b c w h', w=width, h=height)
            out = out * (1 + torch.sigmoid(geo_embed_reshaped))
            
        out = out + feat0
        return out


class I2P_CrossAttention(nn.Module):
    """
    func: Image to Point Cross Attention
    inputs:
        in_dim: input dimension
        out_dim: output dimension
    """

    def __init__(self, in_dim, out_dim):
        super(I2P_CrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim
        self.query_conv = nn.Linear(in_dim, out_dim)
        self.key_conv = nn.Linear(in_dim, out_dim)
        self.value_conv = nn.Linear(in_dim, out_dim)
        self.out_conv = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat0, feat1):
        '''
        :param feat0: query (image features) size:[B * C * N]
        :param feat1: key & value (point cloud features) size:[B * C * W * H]
        :return: attentional value
        '''
        _, C1, N = feat0.size()
        batch_size, C, width, height = feat1.size()
        assert C1 == C

        # 处理图像特征
        proj_query = self.query_conv(rearrange(feat0, 'b c n -> b n c'))

        # 处理点云特征
        proj_key = self.key_conv(rearrange(feat1, 'b c w h -> b (w h) c'))
        proj_value = self.value_conv(rearrange(feat1, 'b c w h -> b (w h) c'))

        # 计算注意力
        energy = torch.bmm(proj_query, rearrange(proj_key, 'b wh c -> b c wh'))
        attention = self.softmax(energy) / C ** .5

        # 应用注意力
        out = self.out_conv(torch.bmm(attention, proj_value))
        out = rearrange(out, 'b n c -> b c n')
        out = out + feat0
        return out


class SelfAttention(nn.Module):
    """
    func: Self Attention with geometric information for point cloud
    """

    def __init__(self, in_dim, out_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # 添加几何信息相关的层
        self.geo_proj = nn.Linear(out_dim, out_dim)
        self.geo_gate = nn.Linear(out_dim * 2, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, geo_embed=None):
        """
        inputs:
            x: input feature maps (B X C X W X H)
            geo_embed: geometric embedding
        returns:
            out: self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        # 处理输入特征
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        
        # 如果提供了几何信息，将其融合到查询中
        if geo_embed is not None:
            geo_embed = self.geo_proj(geo_embed)
            gate = torch.sigmoid(self.geo_gate(torch.cat([proj_query, geo_embed], dim=-1)))
            proj_query = proj_query + gate * geo_embed

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        # 计算注意力
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # 如果提供了几何信息，调制输出
        if geo_embed is not None:
            geo_embed_reshaped = geo_embed.view(m_batchsize, C, width, height)
            out = out * (1 + torch.sigmoid(geo_embed_reshaped))

        return out, attention


if __name__ == '__main__':
    x = torch.randn(size=(4, 16, 20, 20))
    self_atten_spatial = SelfAttention(16, 4)
    y = self_atten_spatial(x)
    print('y.size:', y[0].size())

'''
y.size: torch.Size([4, 16, 20, 20])
'''
