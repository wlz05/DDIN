# DDIN: Domain-Aware Disentanglement Interaction Network for Multimodal Fake News Detection

"""
DDIN.py - DDIN: Domain-Aware Disentanglement Interaction Network
整合版本 - 以第二份代码为准，第一份代码以注释形式保留
"""

import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from copy import deepcopy
import math


# =========================================================================
# FGM 对抗训练类
# =========================================================================
class FGM():
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='bert.embeddings.word_embeddings.weight'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# =========================================================================
# EMA (Exponential Moving Average) 类
# =========================================================================
class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# =========================================================================
# Warmup + Cosine Annealing 学习率调度器
# =========================================================================
class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.base_lrs[i] * (self.current_epoch / self.warmup_epochs)
                param_group['lr'] = lr
        else:
            for i, param_group in enumerate(self.optimizer.param_groups):
                progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                lr = self.eta_min + (self.base_lrs[i] - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                param_group['lr'] = lr


# =========================================================================
# 多尺度语义投影层 (Multi-Scale Semantic Projection Layer)
# =========================================================================
class MultiScaleSemanticProjection(nn.Module):
    """
    多尺度语义投影层 - 通过多个并行投影通道捕获多义性
    """

    def __init__(self, input_dim, output_dim, num_scales=3):
        super(MultiScaleSemanticProjection, self).__init__()
        self.num_scales = num_scales
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_scales)
        ])
        self.fusion = nn.Linear(output_dim * num_scales, output_dim)

    def forward(self, x):
        scale_features = [proj(x) for proj in self.projections]
        fused = torch.cat(scale_features, dim=-1)
        return self.fusion(fused)


# =========================================================================
# 层次化冲突协同感知网络 (Hierarchical Conflict Synergy Network)
# =========================================================================
class HierarchicalConflictSynergy(nn.Module):
    """
    层次化冲突协同机制 - 让不同粒度的冲突特征相互通信
    """

    def __init__(self, hidden_dim, num_heads=4):
        super(HierarchicalConflictSynergy, self).__init__()
        self.transformer_block = Block(
            dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )

    def forward(self, conflict_ll, conflict_gl, conflict_gg):
        # 将三种冲突特征堆叠为序列
        conflicts = torch.stack([conflict_ll, conflict_gl, conflict_gg], dim=1)  # [B, 3, D]
        # 通过Transformer进行冲突传播
        synergized = self.transformer_block(conflicts)  # [B, 3, D]
        return synergized[:, 0], synergized[:, 1], synergized[:, 2]


# =========================================================================
# 领域自适应不一致性加权模块 (Domain-Adaptive Inconsistency Weighting)
# =========================================================================
class DomainAdaptiveWeighting(nn.Module):
    """
    领域自适应加权 - 根据领域动态调整不同冲突的重要性
    """

    def __init__(self, num_domains, hidden_dim):
        super(DomainAdaptiveWeighting, self).__init__()
        self.domain_embeddings = nn.Embedding(num_domains, hidden_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3个门控值对应LL, GL, GG
            nn.Softmax(dim=-1)
        )

    def forward(self, domain_ids, conflict_ll, conflict_gl, conflict_gg):
        # 获取领域嵌入
        domain_emb = self.domain_embeddings(domain_ids)  # [B, D]
        # 生成门控权重
        gates = self.gate_network(domain_emb)  # [B, 3]
        # 加权融合
        weighted_conflict = (
                gates[:, 0:1] * conflict_ll +
                gates[:, 1:2] * conflict_gl +
                gates[:, 2:3] * conflict_gg
        )
        return weighted_conflict, gates


# =========================================================================
# DDIN 主模型 (DDIN Model)
# =========================================================================
class DDIN(nn.Module):
    """
    DDIN: Domain-Aware Disentanglement Interaction Network
    多模态虚假新闻检测模型
    """

    def __init__(self, bert_model_path, mae_model_path, clip_model_name,
                 num_domains=9, hidden_dim=768, num_classes=2):
        super(DDIN, self).__init__()

        # ===== (a) 双流多粒度特征提取 =====
        # BERT for local text features
        self.bert = BertModel.from_pretrained(bert_model_path)

        # MAE for local image features
        self.mae = models_mae.__dict__['mae_vit_base_patch16'](norm_pix_loss=False)
        checkpoint = torch.load(mae_model_path, map_location='cpu')
        self.mae.load_state_dict(checkpoint['model'], strict=False)

        # CLIP for global features
        self.clip_model, _ = load_from_name(clip_model_name, download_root='./model_weights/clip_cn/')

        # ===== (b) 多尺度语义投影层 =====
        self.text_local_proj = MultiScaleSemanticProjection(768, hidden_dim, num_scales=3)
        self.image_local_proj = MultiScaleSemanticProjection(768, hidden_dim, num_scales=3)
        self.text_global_proj = MultiScaleSemanticProjection(512, hidden_dim, num_scales=3)
        self.image_global_proj = MultiScaleSemanticProjection(512, hidden_dim, num_scales=3)

        # ===== (c) 多粒度跨模态不一致性挖掘 =====
        # Global-Global Inconsistency
        self.gg_inconsistency = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Local-Local Inconsistency (Cross-Attention)
        self.ll_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.ll_inconsistency = nn.Sequential(
            nn.Conv1d(2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Global-Local Inconsistency
        self.gl_cross_transformer = Block(
            dim=hidden_dim,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )

        # ===== (d) 层次化冲突协同感知网络 =====
        self.conflict_synergy = HierarchicalConflictSynergy(hidden_dim, num_heads=4)

        # ===== (e) 领域自适应不一致性加权与融合 =====
        self.domain_weighting = DomainAdaptiveWeighting(num_domains, hidden_dim)

        # ===== (f) 领域自适应多模态全局融合 =====
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 分类器
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # 辅助分类器 (用于多任务学习)
        self.fusion_classifier = nn.Linear(hidden_dim, num_classes)
        self.image_classifier = nn.Linear(hidden_dim, num_classes)
        self.text_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, content, content_masks, comments, comments_masks,
                content_emotion, comments_emotion, emotion_gap, style_features,
                image, category, **kwargs):
        batch_size = content.size(0)

        # ===== 特征提取 =====
        # Local text features (BERT)
        bert_output = self.bert(content, attention_mask=content_masks)
        text_local = bert_output.last_hidden_state  # [B, L, 768]

        # Local image features (MAE)
        with torch.no_grad():
            mae_output = self.mae.forward_encoder(image, mask_ratio=0)
            image_local = mae_output[0]  # [B, P, 768]

        # Global features (CLIP)
        text_global = self.clip_model.encode_text(content)  # [B, 512]
        image_global = self.clip_model.encode_image(image)  # [B, 512]

        # ===== 多尺度语义投影 =====
        text_local_proj = self.text_local_proj(text_local.mean(dim=1))  # [B, D]
        image_local_proj = self.image_local_proj(image_local.mean(dim=1))  # [B, D]
        text_global_proj = self.text_global_proj(text_global)  # [B, D]
        image_global_proj = self.image_global_proj(image_global)  # [B, D]

        # ===== 多粒度跨模态不一致性挖掘 =====
        # 1. Global-Global Inconsistency
        gg_concat = torch.cat([
            text_global_proj,
            image_global_proj,
            text_global_proj - image_global_proj
        ], dim=-1)
        conflict_gg = self.gg_inconsistency(gg_concat)  # [B, D]

        # 2. Local-Local Inconsistency
        attn_output, attn_weights = self.ll_cross_attn(
            text_local, image_local, image_local
        )
        row_max = attn_weights.max(dim=-1)[0]  # [B, L]
        col_max = attn_weights.max(dim=-2)[0]  # [B, P]
        # 简化处理：取平均
        ll_features = torch.stack([row_max.mean(dim=1), col_max.mean(dim=1)], dim=1)  # [B, 2]
        conflict_ll = self.ll_inconsistency(ll_features).squeeze(-1)  # [B, D]

        # 3. Global-Local Inconsistency
        gl_concat = torch.stack([text_global_proj, image_local_proj], dim=1)  # [B, 2, D]
        conflict_gl = self.gl_cross_transformer(gl_concat).mean(dim=1)  # [B, D]

        # ===== 层次化冲突协同 =====
        conflict_ll_syn, conflict_gl_syn, conflict_gg_syn = self.conflict_synergy(
            conflict_ll, conflict_gl, conflict_gg
        )

        # ===== 领域自适应加权 =====
        adaptive_conflict, gate_weights = self.domain_weighting(
            category, conflict_ll_syn, conflict_gl_syn, conflict_gg_syn
        )

        # ===== 全局融合 =====
        fusion_input = torch.cat([
            text_global_proj,
            image_global_proj,
            adaptive_conflict,
            text_global_proj * image_global_proj  # 交互特征
        ], dim=-1)
        fused_features = self.final_fusion(fusion_input)  # [B, D]

        # ===== 分类 =====
        final_pred = self.classifier(fused_features)
        fusion_pred = self.fusion_classifier(adaptive_conflict)
        image_pred = self.image_classifier(image_global_proj)
        text_pred = self.text_classifier(text_global_proj)

        return final_pred, fusion_pred, image_pred, text_pred, \
            fused_features, adaptive_conflict, text_global_proj, image_global_proj


# =========================================================================
# 对比损失 (Contrastive Loss)
# =========================================================================
class AdaptiveContrastiveLoss(nn.Module):
    """
    自适应对比损失 - 用于增强跨模态一致性学习
    """

    def __init__(self, temperature=0.07):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, text_features, image_features, labels):
        # 归一化
        text_features = nn.functional.normalize(text_features, dim=-1)
        image_features = nn.functional.normalize(image_features, dim=-1)

        # 计算相似度矩阵
        similarity = torch.matmul(text_features, image_features.T) / self.temperature

        # 构建标签矩阵
        labels = labels.view(-1, 1)
        label_matrix = (labels == labels.T).float()

        # 对比损失
        loss = -torch.log(
            torch.exp(similarity * label_matrix).sum(dim=1) /
            torch.exp(similarity).sum(dim=1)
        ).mean()

        return loss


# =========================================================================
# Trainer 类
# =========================================================================
class Trainer():
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict,
                 weight_decay, save_param_dir, early_stop, epoches,
                 mae_model_path='./model_weights/mae_pretrain_vit_base.pth',
                 clip_model_name='ViT-B-16',
                 use_fgm=True, use_ema=True,
                 contrastive_weight=0.1, logger=None):

        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.save_param_dir = save_param_dir
        self.use_fgm = use_fgm
        self.use_ema = use_ema
        self.contrastive_weight = contrastive_weight

        # 初始化模型
        num_domains = len(category_dict)
        self.model = DDIN(
            bert_model_path=bert,
            mae_model_path=mae_model_path,
            clip_model_name=clip_model_name,
            num_domains=num_domains,
            hidden_dim=emb_dim,
            num_classes=mlp_dims[-1]
        )

        if self.use_cuda:
            self.model = self.model.cuda()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_loss = AdaptiveContrastiveLoss()

        # 优化器 - 分层学习率
        bert_params = list(self.model.bert.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if 'bert' not in n]

        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': lr * 0.1},
            {'params': other_params, 'lr': lr}
        ], weight_decay=weight_decay)

        # 学习率调度器
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=3,
            max_epochs=epoches,
            eta_min=1e-6
        )

    def train(self, logger=None):
        if logger:
            logger.info('Start training...')

        # 初始化FGM和EMA
        fgm = FGM(self.model, epsilon=0.5) if self.use_fgm else None
        ema = EMA(self.model, decay=0.999) if self.use_ema else None

        recorder = Recorder(self.early_stop)

        for epoch in range(self.epoches):
            self.model.train()
            avg_loss = Averager()
            avg_cls_loss = Averager()
            avg_con_loss = Averager()
            avg_adaptive_weight = Averager()

            train_iter = tqdm.tqdm(self.train_loader)
            for step, batch in enumerate(train_iter):
                batch_data = clipdata2gpu(batch) if self.use_cuda else batch
                batch_label = batch_data['label']

                self.optimizer.zero_grad()

                # 前向传播
                final_pred, fusion_pred, image_pred, text_pred, \
                    fused_feat, adaptive_conflict, text_feat, image_feat = self.model(**batch_data)

                # 分类损失
                cls_loss = (
                        self.criterion(final_pred, batch_label) +
                        0.5 * self.criterion(fusion_pred, batch_label) +
                        0.3 * self.criterion(image_pred, batch_label) +
                        0.3 * self.criterion(text_pred, batch_label)
                )

                # 自适应对比损失
                adaptive_con_loss = self.contrastive_loss(text_feat, image_feat, batch_label)
                batch_adaptive_weight = torch.tensor([self.contrastive_weight]).to(batch_label.device)

                # 总损失
                loss = cls_loss + batch_adaptive_weight * adaptive_con_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # FGM对抗训练
                if self.use_fgm and fgm:
                    fgm.attack()
                    final_pred_adv, fusion_pred_adv, image_pred_adv, text_pred_adv, \
                        _, _, text_feat_adv, image_feat_adv = self.model(**batch_data)

                    cls_loss_adv = (
                            self.criterion(final_pred_adv, batch_label) +
                            0.5 * self.criterion(fusion_pred_adv, batch_label) +
                            0.3 * self.criterion(image_pred_adv, batch_label) +
                            0.3 * self.criterion(text_pred_adv, batch_label)
                    )
                    adaptive_con_loss_adv = self.contrastive_loss(
                        text_feat_adv, image_feat_adv, batch_label
                    )
                    loss_adv = cls_loss_adv + batch_adaptive_weight * adaptive_con_loss_adv
                    loss_adv.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    fgm.restore()

                self.optimizer.step()
                if self.use_ema and ema:
                    ema.update()

                avg_loss.add(loss.item())
                avg_cls_loss.add(cls_loss.item())
                avg_con_loss.add(adaptive_con_loss.item())
                avg_adaptive_weight.add(batch_adaptive_weight.mean().item())

            self.scheduler.step()
            current_lr_bert = self.optimizer.param_groups[0]['lr']
            current_lr_other = self.optimizer.param_groups[1]['lr']
            print(
                'Training Epoch {}; Total Loss: {:.4f}; Cls Loss: {:.4f}; Adaptive Con Loss: {:.4f}; Avg Weight: {:.4f}; LR_BERT: {:.2e}; LR_Other: {:.2e}'.format(
                    epoch + 1, avg_loss.item(), avg_cls_loss.item(), avg_con_loss.item(), avg_adaptive_weight.item(),
                    current_lr_bert, current_lr_other))

            if self.use_ema and ema:
                ema.apply_shadow()
            results0, results1, results2, results3 = self.test(self.val_loader)
            if self.use_ema and ema:
                ema.restore()

            mark = recorder.add(results0)
            if mark == 'save':
                if self.use_ema and ema:
                    ema.apply_shadow()
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, 'parameter_DDIN.pkl'))
                if self.use_ema and ema:
                    ema.restore()
            elif mark == 'esc':
                break
            else:
                continue

        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_DDIN.pkl')))
        if self.use_ema and ema:
            ema.apply_shadow()
        results0, results1, results2, results3 = self.test(self.test_loader)
        if logger:
            logger.info("start testing......")
            logger.info("test score: {}.".format(results0))
            logger.info("lr: {}, contrastive_weight: {}, avg test score: {}.\n\n".format(
                self.lr, self.contrastive_weight, results0))
        print(results0)
        return results0, results1, results2, results3

    def test(self, dataloader):
        pred0 = []
        pred1 = []
        pred2 = []
        pred3 = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch) if self.use_cuda else batch
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                final_label_pred_list, fusion_label_pred_list, image_label_pred_list, text_label_pred_list, _, _, _, _ = self.model(
                    **batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred0.extend(final_label_pred_list.detach().cpu().numpy().tolist())
                pred1.extend(fusion_label_pred_list.detach().cpu().numpy().tolist())
                pred2.extend(image_label_pred_list.detach().cpu().numpy().tolist())
                pred3.extend(text_label_pred_list.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred0, category, self.category_dict), \
            metrics(label, pred1, category, self.category_dict), \
            metrics(label, pred2, category, self.category_dict), \
            metrics(label, pred3, category, self.category_dict)

mae_model_path=self.config.get('mae_model_path', './model_weights/mae_pretrain_vit_base.pth'),
clip_model_name=self.config.get('cn_clip_model_name', 'ViT-B-16'),
use_fgm=self.config.get('use_fgm', True),
use_ema=self.config.get('use_ema', True),
contrastive_weight=self.config.get('contrastive_weight', 0.1),
logger=logger
)


# =========================================================================
# 第一份代码的注释版本 (保留原有思路)
# =========================================================================
"""
# 原版代码思路 (run.py):
# 
# 1. 数据加载器设计:
#    - GossipCop: 使用 FakeNet_dataset + collate_fn_gossipcop
#    - Weibo: 使用 utils.weibo_clip_dataloader.bert_data
#    - Weibo21: 使用 utils.weibo21_clip_dataloader.bert_data
#
# 2. Trainer选择:
#    - GossipCop: model.domain_gossipcop.Trainer
#    - Weibo/Weibo21: model.domain_weibo.Trainer
#
# 3. 配置管理:
#    - 通过config字典传递所有参数
#    - 支持不同数据集的路径配置
#    - 支持BERT/CLIP模型路径配置
#
# 4. 训练流程:
#    - Run类负责数据加载和Trainer初始化
#    - Trainer类负责具体的训练和测试
#    - 支持early stopping和模型保存
#
# 5. 关键特性:
#    - Logger日志记录
#    - 异常处理和错误提示
#    - 模块化设计便于扩展
#    - 支持多数据集和多模型
"""