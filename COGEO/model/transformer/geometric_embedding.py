import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码，用于将距离和角度索引转换为嵌入向量"""
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        # 计算 div_term 时，确保 d_model/2 是整数
        half_d_model = d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, dtype=torch.float) * (-np.log(10000.0) / half_d_model))
        self.register_buffer('div_term', div_term) # Shape: [d_model/2]

    def forward(self, x):
        """
        Args:
            x: Tensor, containing the positions (e.g., distances or angles).
               Can have shape [N], [N, N], [N, K, K], etc.
               Assumes the values to be encoded are directly in x.
        """
        # 确定 position 张量 - 直接使用输入 x
        position = x.unsqueeze(-1).float() # 添加一个维度用于广播, shape [..., 1]

        # 初始化 pe 张量，确保最后一个维度是 d_model
        # 保留 x 原有的维度，并在最后添加 d_model 维度
        pe_shape = x.shape + (self.d_model,)
        pe = torch.zeros(pe_shape, device=x.device) # shape [..., d_model]

        # 计算正弦和余弦编码
        # div_term shape is [d_model/2]
        # position * self.div_term 会广播: [..., 1] * [d_model/2] -> [..., d_model/2]
        sin_component = torch.sin(position * self.div_term) # shape [..., d_model/2]
        cos_component = torch.cos(position * self.div_term) # shape [..., d_model/2]

        # 检查 sin/cos component 的形状是否正确
        # expected_last_dim = self.d_model // 2
        # if sin_component.shape[-1] != expected_last_dim:
        #     raise RuntimeError(f"Shape mismatch: sin component last dim {sin_component.shape[-1]}, expected {expected_last_dim}")
        # if cos_component.shape[-1] != expected_last_dim:
        #     raise RuntimeError(f"Shape mismatch: cos component last dim {cos_component.shape[-1]}, expected {expected_last_dim}")


        # 填充 pe 张量
        pe[..., 0::2] = sin_component
        pe[..., 1::2] = cos_component

        return pe



class GeometricStructureEmbedding(nn.Module):
    """几何结构嵌入模块，计算点云中点之间的距离和角度，并将其编码为嵌入向量"""
    def __init__(self, hidden_dim, sigma_d, sigma_a, angle_k, reduction_a='max'):
        super(GeometricStructureEmbedding, self).__init__()
        self.sigma_d = sigma_d
        self.sigma_a = sigma_a
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = angle_k

        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)

        self.reduction_a = reduction_a
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @torch.no_grad()
    def get_embedding_indices(self, points):
        """计算点对之间的距离和三元组角度的索引
        Args:
            points: torch.Tensor (N, 3)
        Returns:
            indices_d: torch.Tensor (N, N)
            indices_a: torch.Tensor (N, K, K)
        """
        # 计算点对距离
        n = points.shape[0]
        dists = torch.sqrt(torch.sum(torch.square(points.unsqueeze(1) - points.unsqueeze(0)), dim=-1))
        indices_d = torch.clamp(torch.round(dists / self.sigma_d), 0, 255).long()

        # 计算三元组角度
        _, indices = torch.topk(dists, self.angle_k, dim=1, largest=False)
        vectors = points[indices] - points.unsqueeze(1)
        vectors_norm = torch.linalg.norm(vectors, dim=-1)
        vectors_unit = vectors / (vectors_norm.unsqueeze(-1) + 1e-8)
        angles = torch.acos(
            torch.clamp(torch.sum(vectors_unit.unsqueeze(2) * vectors_unit.unsqueeze(1), dim=-1), -1.0, 1.0)
        )
        indices_a = torch.clamp(torch.round(angles * self.factor_a), 0, 255).long()

        return indices_d, indices_a

    def forward(self, points):
        """计算几何结构嵌入
        Args:
            points: torch.Tensor (B, N, 3)
        Returns:
            embeddings: torch.Tensor (B, N, C)
        """
        batch_size, n_points, _ = points.shape
        embeddings = []
        for i in range(batch_size):
            # 计算距离和角度索引
            indices_d, indices_a = self.get_embedding_indices(points[i])
            
            # 距离嵌入
            embed_d = self.embedding(indices_d)
            embed_d = torch.mean(self.proj_d(embed_d), dim=1)
            
            # 角度嵌入
            embed_a = self.embedding(indices_a)
            embed_a = self.proj_a(embed_a)
            if self.reduction_a == 'max':
                embed_a = torch.max(embed_a.view(n_points, -1, embed_a.shape[-1]), dim=1)[0]
            else:
                embed_a = torch.mean(embed_a.view(n_points, -1, embed_a.shape[-1]), dim=1)
            
            # 合并距离和角度嵌入
            embedding = embed_d + embed_a
            embeddings.append(embedding)

        return torch.stack(embeddings)  # (B, N, C)