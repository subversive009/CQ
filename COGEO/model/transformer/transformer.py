import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import LinearAttention, FullAttention
#导入集合嵌入模块
from .geometric_embedding import GeometricStructureEmbedding
import math

# D_MODEL = 128
# D_FFN = 256
# NHEAD = 4
# LAYER_NAMES = ['self', 'cross'] * 4
# ATTENTION = 'full'  # options: ['linear', 'full']


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='full'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # 多头注意力层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 几何信息投影层
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_k_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 注意力机制
        if attention == 'linear':
            self.attention = LinearAttention()
        else:
            self.attention = FullAttention()
        
        # 输出投影
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        
        # 可学习的残差权重
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

    def forward(self, x, source, geo_embed=None, is_pc_self_attn=False, is_cross_attn=False):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            geo_embed (torch.Tensor, optional): 几何嵌入 [N, L, C]
            is_pc_self_attn (bool): 是否为点云自注意力
            is_cross_attn (bool): 是否为交叉注意力
        """
        bs = x.size(0)
        identity = x
        
        # Pre-LN
        x_norm = self.norm_q(x)
        source_norm = self.norm_k(source)
        
        # 投影查询、键、值
        q = self.q_proj(x_norm)
        k = self.k_proj(source_norm)
        v = self.v_proj(self.norm_v(source))

        # 重塑为多头形式
        q = q.view(bs, -1, self.nhead, self.dim).permute(0, 2, 1, 3)  # [N, H, L, D]
        k = k.view(bs, -1, self.nhead, self.dim).permute(0, 2, 1, 3)  # [N, H, S, D]
        v = v.view(bs, -1, self.nhead, self.dim).permute(0, 2, 1, 3)  # [N, H, S, D]

        if geo_embed is not None:
            g = self.g_proj(geo_embed)
            g = g.view(bs, -1, self.nhead, self.dim).permute(0, 2, 1, 3)  # [N, H, L, D]
            
            if is_pc_self_attn or is_cross_attn:
                # 计算几何注意力得分
                geo_attn = torch.einsum('bhld,bhld->bhl', q, g) / math.sqrt(self.dim)
                geo_attn = F.softmax(geo_attn, dim=-1)
                q = q + geo_attn.unsqueeze(-1) * g
            else:
                # 图像到点云的交叉注意力：增强键值对
                g_k = self.g_k_proj(geo_embed)
                g_k = g_k.view(bs, -1, self.nhead, self.dim).permute(0, 2, 1, 3)
                k = k + g_k
                v = v + g_k

        # 调整张量维度以适应 LinearAttention/FullAttention 的输入格式
        q = q.permute(0, 2, 1, 3)  # [N, L, H, D]
        k = k.permute(0, 2, 1, 3)  # [N, S, H, D]
        v = v.permute(0, 2, 1, 3)  # [N, S, H, D]
        
        # 使用 LinearAttention 或 FullAttention 计算注意力
        message = self.attention(q, k, v)  # [N, L, H, D]
        
        # 重塑回原始维度
        message = message.reshape(bs, -1, self.nhead * self.dim)  # [N, L, C]
        message = self.merge(message)
        
        # 第一个残差连接
        x = identity + self.gamma1 * message

        # FFN
        message = self.mlp(self.norm2(x))
        
        # 第二个残差连接
        x = x + self.gamma2 * message

        return x


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, D_MODEL, NHEAD, LAYER_NAMES, ATTENTION, 
                 use_geo_embedding=False, geo_config=None):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = D_MODEL
        self.nhead = NHEAD
        self.layer_names = LAYER_NAMES
        
        # 创建编码器层
        self.self_layers = nn.ModuleList()
        self.cross_layers = nn.ModuleList()
        
        for name in self.layer_names:
            if name == 'self':
                self.self_layers.append(LoFTREncoderLayer(D_MODEL, NHEAD, ATTENTION))
            elif name == 'cross':
                self.cross_layers.append(LoFTREncoderLayer(D_MODEL, NHEAD, ATTENTION))
        
        # 几何嵌入相关配置
        self.use_geo_embedding = use_geo_embedding
        if self.use_geo_embedding:
            if geo_config is None:
                # 默认配置
                geo_config = {
                    'hidden_dim': D_MODEL,  # 确保与模型维度一致
                    'sigma_d': 0.2,  # 距离量化参数
                    'sigma_a': 15,   # 角度量化参数
                    'angle_k': 3     # 角度计算中的邻居数
                }
            # 确保传递正确的hidden_dim参数
            self.geometric_embedding = GeometricStructureEmbedding(
                hidden_dim=geo_config['hidden_dim'],
                sigma_d=geo_config['sigma_d'],
                sigma_a=geo_config['sigma_a'],
                angle_k=geo_config['angle_k']
            )
        
        # 输入输出归一化
        self.input_norm = nn.LayerNorm(D_MODEL)
        self.output_norm = nn.LayerNorm(D_MODEL)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_pc, feat_img, pc_coords=None):
        """
        Args:
            feat_pc (torch.Tensor): 点云特征 [N, L, C]
            feat_img (torch.Tensor): 图像特征 [N, S, C]
            pc_coords (torch.Tensor, optional): 点云坐标 [N, L, 3]
        """
        assert self.d_model == feat_pc.size(2), "the feature number of src and transformer must be equal"
        
        # 输入归一化
        feat_pc = self.input_norm(feat_pc)
        feat_img = self.input_norm(feat_img)
        
        # 计算几何嵌入
        geo_embed = None
        if self.use_geo_embedding and pc_coords is not None:
            geo_embed = self.geometric_embedding(pc_coords)

        # 分别处理自注意力和交叉注意力层
        for layer_idx, self_layer in enumerate(self.self_layers):
            # 使用渐进式的梯度检查点策略
            if layer_idx >= len(self.self_layers) // 2:  # 只在后半部分层使用
                feat_pc = torch.utils.checkpoint.checkpoint(
                    self_layer, feat_pc, feat_pc, geo_embed, True, False
                )
                feat_img = torch.utils.checkpoint.checkpoint(
                    self_layer, feat_img, feat_img, None, False, False
                )
            else:
                feat_pc = self_layer(feat_pc, feat_pc, geo_embed, True, False)
                feat_img = self_layer(feat_img, feat_img, None, False, False)
            
        for cross_layer in self.cross_layers:
            # 点云到图像的交叉注意力
            feat_pc_new = torch.utils.checkpoint.checkpoint(
                cross_layer, feat_pc, feat_img, geo_embed, False, True
            )
            # 图像到点云的交叉注意力
            feat_img_new = torch.utils.checkpoint.checkpoint(
                cross_layer, feat_img, feat_pc, None, False, False
            )
            feat_pc, feat_img = feat_pc_new, feat_img_new

        # 输出归一化
        feat_pc = self.output_norm(feat_pc)
        feat_img = self.output_norm(feat_img)

        return feat_pc, feat_img