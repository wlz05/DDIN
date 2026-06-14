# DDIN: Domain-Aware Disentanglement Interaction Network for Multimodal Fake News Detection

import torch
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Function


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_Mu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP_Mu, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 9))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class clip_fuion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(clip_fuion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnn_extractor(torch.nn.Module):
    def __init__(self, input_size, feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(input_size, feature_num, kernel)
                                          for kernel, feature_num in feature_kernel.items()]
                                         )

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1)
        feature = [conv(input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class image_cnn_extractor(nn.Module):
    def __init__(self):
        super(image_cnn_extractor, self).__init__()
        self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 24 * 96, 512)
        self.fc2 = nn.Linear(512, 320)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 24 * 96)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class image_extractor(torch.nn.Module):
    def __init__(self, out_channels):
        super(image_extractor, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1, self.img_backbone.bn1, self.img_backbone.relu,
            self.img_backbone.layer1, self.img_backbone.layer2, self.img_backbone.layer3, self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out


class classifier(torch.nn.Module):
    def __init__(self, out_dim=1):
        super(classifier, self).__init__()
        self.trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
        )
        self.classifier1 = nn.Sequential(nn.Linear(64, out_dim))

    def forward(self, x):
        x = self.classifier1(self.trim(x))
        return x


class MaskAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim, 1)

    def forward(self, input, mask):
        score = self.Line(input).view(-1, input.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score, input).squeeze(1)
        return output


class TokenAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores


class Attention(torch.nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x), attn


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# =========================================================================
# DDIN 核心组件
# =========================================================================

class MultiScaleProjectionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_scales=3):
        super(MultiScaleProjectionLayer, self).__init__()
        self.num_scales = num_scales
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_scales)
        ])
        self.fuse = nn.Linear(out_dim * num_scales, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        parts = [self.act(proj(x)) for proj in self.projections]
        concat = torch.cat(parts, dim=-1)
        return self.fuse(concat)


class GlobalGlobalInconsistency(nn.Module):
    def __init__(self, dim):
        super(GlobalGlobalInconsistency, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, f_t_glo, f_i_glo):
        diff = f_t_glo - f_i_glo
        fused = torch.cat([f_t_glo, f_i_glo, diff], dim=-1)
        return self.mlp(fused)


class LocalLocalInconsistency(nn.Module):
    """升级版：使用深度Cross-Attention替代BMM"""

    def __init__(self, text_len, img_len, dim, num_heads=4, num_layers=2):
        super(LocalLocalInconsistency, self).__init__()
        # 双向交叉注意力
        self.text_to_img_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.img_to_text_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        # LayerNorm for stability
        self.text_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.img_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        # 冲突聚合
        self.conflict_aggregator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, f_t_loc, f_i_loc):
        # f_t_loc: (batch, 197, dim), f_i_loc: (batch, 197, dim)

        # 多层交叉注意力交互
        t_out = f_t_loc
        i_out = f_i_loc

        for t2i_attn, i2t_attn, t_norm, i_norm in zip(
                self.text_to_img_attn, self.img_to_text_attn,
                self.text_norms, self.img_norms
        ):
            # Text attends to Image
            t_attn_out, _ = t2i_attn(t_out, i_out, i_out)
            t_out = t_norm(t_out + t_attn_out)

            # Image attends to Text
            i_attn_out, _ = i2t_attn(i_out, t_out, t_out)
            i_out = i_norm(i_out + i_attn_out)

        # 池化为全局冲突特征
        t_conflict = t_out.mean(dim=1)  # (batch, dim)
        i_conflict = i_out.mean(dim=1)  # (batch, dim)

        # 融合双向冲突
        conflict = torch.cat([t_conflict, i_conflict], dim=-1)
        return self.conflict_aggregator(conflict)


class GlobalLocalInconsistency(nn.Module):
    """高阶非线性交互：引入哈达玛积捕获二阶矛盾特征"""

    def __init__(self, dim):
        super(GlobalLocalInconsistency, self).__init__()
        self.attn_T = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.attn_I = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        # 修改：输入维度从 dim*2 改为 dim*3，以容纳哈达玛积
        self.fc = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, f_t_loc, f_i_loc, f_t_glo, f_i_glo):
        # 全局-局部交叉注意力
        out_T, _ = self.attn_T(f_i_glo.unsqueeze(1), f_t_loc, f_t_loc)
        out_I, _ = self.attn_I(f_t_glo.unsqueeze(1), f_i_loc, f_i_loc)

        out_T = out_T.squeeze(1)  # (batch, dim)
        out_I = out_I.squeeze(1)  # (batch, dim)

        # 高阶交互：拼接原始特征 + 哈达玛积（element-wise multiplication）
        hadamard = out_T * out_I  # (batch, dim)
        fused = torch.cat([out_T, out_I, hadamard], dim=-1)  # (batch, dim*3)

        return self.fc(fused)


class HierarchicalConflictSynergy(nn.Module):
    def __init__(self, dim, num_layers=1):
        super(HierarchicalConflictSynergy, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, C_LL, C_GL, C_GG):
        x = torch.stack([C_LL, C_GL, C_GG], dim=1)
        out = self.transformer(x)
        return out


class DomainAwareGating(nn.Module):
    def __init__(self, dim):
        super(DomainAwareGating, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3)
        )

    def forward(self, domain_emb):
        logits = self.mlp(domain_emb)
        return torch.softmax(logits, dim=-1)


# =========================================================================
# Supervised Contrastive Loss
# =========================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Reference: https://arxiv.org/abs/2004.11362
    """

    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features_text, features_image, labels):
        """
        Args:
            features_text: (batch_size, dim) - 文本全局特征
            features_image: (batch_size, dim) - 图像全局特征
            labels: (batch_size,) - 真实标签 (0/1)
        """
        batch_size = features_text.shape[0]

        # L2 normalize
        features_text = F.normalize(features_text, dim=1)
        features_image = F.normalize(features_image, dim=1)

        # 拼接文本和图像特征
        features = torch.cat([features_text, features_image], dim=0)  # (2*batch, dim)
        labels = labels.contiguous().view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)  # (2*batch, 1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建mask: 同类为正样本对
        mask = torch.eq(labels, labels.T).float().cuda()

        # 去除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # 计算log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        loss = -mean_log_prob_pos.mean()
        return loss


# =========================================================================
# Focal Loss for Hard Cases
# =========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and focusing on hard examples
    Reference: https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch_size,) - predicted probabilities (after sigmoid)
            targets: (batch_size,) - ground truth labels (0 or 1)
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # p_t: probability of the true class
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()
