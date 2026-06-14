# DDIN: Domain-Aware Disentanglement Interaction Network for Multimodal Fake News Detection

# ./mm/model/domain.py
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, CLIPModel
import logging

# 从 utils.utils 导入所需的函数
# 添加 calculate_metrics 和 metricsTrueFalse 到导入列表
from utils.utils import (
    data2gpu,
    Averager,
    metrics_original_weibo as metrics,  # 保持别名以兼容
    Recorder,
    clipdata2gpu,
    calculate_metrics,  # <--- 新增导入
    metricsTrueFalse  # <--- 新增导入 (Trainer.test 中会用到)
)

logger = logging.getLogger(__name__)


class AdaIN(nn.Module):  # 从 layers.py 移动到这里或确保 layers.py 中有定义
    def __init__(self):
        super().__init__()

    def mu(self, x):
        if x is None: return None
        if x.dim() == 3:
            return torch.mean(x, dim=1)
        elif x.dim() == 2:
            return torch.mean(x, dim=0, keepdim=True)
        else:
            return torch.mean(x)

    def sigma(self, x):
        if x is None: return None
        if x.dim() == 3:
            mu_val = self.mu(x).unsqueeze(1)
            return torch.sqrt(torch.mean((x - mu_val) ** 2, dim=1) + 1e-8)
        elif x.dim() == 2:
            return torch.sqrt(torch.mean((x - self.mu(x)) ** 2, dim=0, keepdim=True) + 1e-8)
        else:
            return torch.std(x) + 1e-8

    def forward(self, x, mu, sigma):
        if x is None or mu is None or sigma is None: return x
        x_dim = x.dim()
        x_mean = self.mu(x)
        x_std = self.sigma(x)

        if x_dim == 3:
            if x_mean.dim() == 2: x_mean = x_mean.unsqueeze(1)
            if x_std.dim() == 2: x_std = x_std.unsqueeze(1)
        x_norm = (x - x_mean) / (x_std + 1e-8)
        if mu.dim() == 2 and x_norm.dim() == 3: mu = mu.unsqueeze(1)
        if sigma.dim() == 2 and x_norm.dim() == 3: sigma = sigma.unsqueeze(1)
        sigma = torch.relu(sigma) + 1e-8
        return sigma * x_norm + mu


try:
    import models_mae
except ImportError:
    logger.error("Failed to import models_mae. Ensure it's in the PYTHONPATH.")
    models_mae = None

HAS_CUSTOM_LAYERS = False
try:
    from .layers import *
    from .pivot import *

    if 'cnn_extractor' in globals() and callable(globals()['cnn_extractor']):
        pass
    HAS_CUSTOM_LAYERS = True
except ImportError:
    logger.warning("Could not import from .layers or .pivot. Using placeholder definitions for some components.")
    if 'MaskAttention' not in globals():
        class MaskAttention(nn.Module):
            def __init__(self, dim):
                super().__init__(); self.dim = dim

            def forward(self, feat, mask):
                if feat is None: return torch.zeros(1, self.dim, device='cpu')
                if mask is None: return torch.mean(feat, dim=1)
                if mask.dim() == 2 and feat.dim() == 3: mask = mask.unsqueeze(-1)
                if feat.shape[0] != mask.shape[0] or feat.shape[1] != mask.shape[1] or \
                        (mask.shape[-1] != 1 and mask.shape[-1] != feat.shape[-1]):
                    logger.warning(
                        f"MaskAttention shape mismatch: feat {feat.shape}, mask {mask.shape}. Using mean pooling.")
                    return torch.mean(feat, dim=1)
                try:
                    if mask.shape[-1] == 1 and feat.shape[-1] == self.dim: mask = mask.expand_as(feat)
                    masked_feat = feat * mask
                    sum_masked_feat = torch.sum(masked_feat, dim=1);
                    sum_mask = torch.sum(mask, dim=1)
                    return sum_masked_feat / (sum_mask.clamp(min=1e-9))
                except Exception as e:
                    logger.error(f"Error in MaskAttention forward: {e}")
                    return torch.mean(feat, dim=1)

    if 'TokenAttention' not in globals():
        class TokenAttention(nn.Module):
            def __init__(self, dim):
                super().__init__();
                self.dim = dim;
                self.attention_weights = nn.Linear(dim, 1)

            def forward(self, feat):
                if feat is None: return torch.zeros(1, self.dim, device='cpu'), None
                if feat.dim() != 3 or feat.shape[2] != self.dim:
                    logger.error(f"TokenAttention shape error. Expected (B, L, {self.dim}), got {feat.shape}")
                    return torch.zeros(feat.shape[0] if feat.dim() > 1 else 1, self.dim,
                                       device=feat.device if hasattr(feat, 'device') else 'cpu'), None
                e = self.attention_weights(feat);
                alpha = torch.softmax(e, dim=1);
                context = torch.bmm(alpha.transpose(1, 2), feat).squeeze(1);
                return context, alpha

    if 'MLP' not in globals():
        class MLP(nn.Module):
            def __init__(self, in_dim, hidden_dims_list, dropout_rate):
                super().__init__();
                layers = [];
                current_dim = in_dim
                if not hidden_dims_list:
                    layers.append(nn.Linear(current_dim, 1))
                else:
                    for h_dim in hidden_dims_list: layers.append(nn.Linear(current_dim, h_dim)); layers.append(
                        nn.ReLU()); layers.append(nn.Dropout(dropout_rate)); current_dim = h_dim
                    layers.append(nn.Linear(current_dim, 1))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

    if 'MLP_fusion' not in globals():
        class MLP_fusion(nn.Module):
            def __init__(self, in_dim, out_dim, hidden_dims_list, dropout_rate):
                super().__init__();
                layers = [];
                current_dim = in_dim
                if not hidden_dims_list:
                    layers.append(nn.Linear(current_dim, out_dim))
                else:
                    for h_dim in hidden_dims_list: layers.append(nn.Linear(current_dim, h_dim)); layers.append(
                        nn.ReLU()); layers.append(nn.Dropout(dropout_rate)); current_dim = h_dim
                    layers.append(nn.Linear(current_dim, out_dim))
                self.network = nn.Sequential(*layers)  # Placeholder defines self.network

            def forward(self, x):
                return self.network(x)

    if 'cnn_extractor' not in globals():
        HAS_CUSTOM_LAYERS = False


        class cnn_extractor(nn.Module):
            def __init__(self, in_dim_token_vector, feature_kernel_config, out_features=320):
                super().__init__()
                self.in_dim = in_dim_token_vector
                if isinstance(feature_kernel_config, dict) and len(feature_kernel_config) == 1 and \
                        list(feature_kernel_config.keys())[0] == 1:
                    self.out_dim = list(feature_kernel_config.values())[0]
                else:
                    self.out_dim = out_features
                self.pool_and_reduce = nn.Sequential(
                    nn.Linear(self.in_dim, self.out_dim),
                    nn.ReLU()
                )

            def forward(self, x_seq):
                if x_seq is None: return torch.zeros(1, self.out_dim, device='cpu')
                if hasattr(x_seq, 'shape') and len(x_seq.shape) > 2 and x_seq.shape[-1] != self.in_dim:
                    logger.warning(
                        f"Placeholder cnn_extractor input dim mismatch. Expected {self.in_dim}, got {x_seq.shape[-1]}.")
                pooled_x = torch.mean(x_seq, dim=1)
                return self.pool_and_reduce(pooled_x)

    if 'LayerNorm' not in globals():
        class LayerNorm(nn.Module):
            def __init__(self, dim, eps=1e-12): super().__init__(); self.norm = nn.LayerNorm(dim, eps=eps)

            def forward(self, x): return self.norm(x) if x is not None else None

    if 'TransformerLayer' not in globals():
        class TransformerLayer(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()

            def forward(self, x, mask=None): return self.fc(x)

    if 'MLP_trans' not in globals():
        class MLP_trans(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()

            def forward(self, x): return self.fc(x)

    if 'SimpleGate' not in globals():
        class SimpleGate(nn.Module):
            def __init__(self, dim=1): super(SimpleGate, self).__init__(); self.dim = dim

            def forward(self, x): x1, x2 = x.chunk(2, dim=self.dim); return x1 * x2

    if 'AdaIN' not in globals():
        class AdaIN(nn.Module):
            def __init__(self):
                super().__init__()

            def mu(self, x):
                if x is None: return None
                if x.dim() == 3:
                    return torch.mean(x, dim=1)
                elif x.dim() == 2:
                    return torch.mean(x, dim=0, keepdim=True)
                else:
                    return torch.mean(x)

            def sigma(self, x):
                if x is None: return None
                if x.dim() == 3:
                    mu_val = self.mu(x)
                    if mu_val.dim() == 1 and x.shape[0] == mu_val.shape[0]:
                        mu_val = mu_val.unsqueeze(1).unsqueeze(-1)
                    elif mu_val.dim() == 2 and x.shape[0] == mu_val.shape[0] and x.shape[2] == mu_val.shape[1]:
                        mu_val = mu_val.unsqueeze(1)
                    return torch.sqrt(torch.mean((x - mu_val) ** 2, dim=1) + 1e-8)
                elif x.dim() == 2:
                    return torch.sqrt(torch.mean((x - self.mu(x)) ** 2, dim=0, keepdim=True) + 1e-8)
                else:
                    return torch.std(x) + 1e-8

            def forward(self, x, mu_style, sigma_style):
                if x is None or mu_style is None or sigma_style is None: return x
                x_dim = x.dim()
                x_mean = self.mu(x)
                x_std = self.sigma(x)
                if x_dim == 3:
                    if x_mean.dim() == 2: x_mean = x_mean.unsqueeze(1)
                    if x_std.dim() == 2: x_std = x_std.unsqueeze(1)
                x_norm = (x - x_mean) / (x_std + 1e-8)
                if mu_style.dim() == x_norm.dim() - 1: mu_style = mu_style.unsqueeze(1)
                if sigma_style.dim() == x_norm.dim() - 1: sigma_style = sigma_style.unsqueeze(1)
                sigma_style = torch.relu(sigma_style) + 1e-8
                return sigma_style * x_norm + mu_style

from timm.models.vision_transformer import Block

try:
    import cn_clip.clip as cn_clip_lib
except ImportError:
    cn_clip_lib = None


class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert_path, clip_path, mae_checkpoint_path,
                 dataset_type,
                 out_channels=320, dropout=0.2, use_cuda=True,
                 text_token_len=197, image_token_len_mae=197,
                 cn_clip_vision_model_name="ViT-B-16",
                 cn_clip_text_model_name="RoBERTa-wwm-ext-base-chinese"
                 ):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.use_cuda = use_cuda
        self.dataset_type = dataset_type
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.num_share = 1
        self.unified_dim = emb_dim
        self.text_dim = 768
        self.image_dim = 768
        self.text_token_len_expected = text_token_len
        self.image_token_len_expected_mae = image_token_len_mae + 1

        self.bert = None
        self.image_model_mae = None
        self.clip_model_hf = None
        self.clip_model_cn = None

        try:
            logger.info(f"Loading BERT from: {bert_path}")
            self.bert = BertModel.from_pretrained(bert_path)
            for param in self.bert.parameters(): param.requires_grad_(False)
            if self.use_cuda: self.bert.cuda()
            logger.info("BERT loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BERT model from {bert_path}: {e}")
            self.bert = None

        try:
            if models_mae is not None:
                model_size_mae = "base"
                self.image_model_mae = models_mae.__dict__[f"mae_vit_{model_size_mae}_patch16"](norm_pix_loss=False)
                if mae_checkpoint_path and os.path.exists(mae_checkpoint_path):
                    logger.info(f"Loading MAE checkpoint from: {mae_checkpoint_path}")
                    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
                    state_dict_key = 'model' if 'model' in checkpoint else None
                    if state_dict_key:
                        load_msg = self.image_model_mae.load_state_dict(checkpoint[state_dict_key], strict=False)
                    else:
                        load_msg = self.image_model_mae.load_state_dict(checkpoint, strict=False)
                    logger.info(f"MAE checkpoint loaded: {load_msg}")
                else:
                    logger.warning(
                        f"MAE checkpoint path not found or not provided: {mae_checkpoint_path}. MAE model uses random weights.")
                for param in self.image_model_mae.parameters(): param.requires_grad_(False)
                if self.use_cuda: self.image_model_mae.cuda()
                logger.info("MAE model initialized.")
            else:
                logger.error("models_mae module not available. MAE model cannot be loaded.")
                self.image_model_mae = None
        except Exception as e:
            logger.error(f"Failed to load MAE model: {e}")
            self.image_model_mae = None

        try:
            if self.dataset_type == 'gossipcop':
                logger.info(f"Loading HuggingFace CLIP model from: {clip_path}")
                self.clip_model_hf = CLIPModel.from_pretrained(clip_path)
                for param in self.clip_model_hf.parameters(): param.requires_grad_(False)
                if self.use_cuda: self.clip_model_hf.cuda()
                logger.info("HuggingFace CLIP model loaded successfully.")
            elif self.dataset_type in ['weibo', 'weibo21']:
                if cn_clip_lib:
                    logger.info(
                        f"Loading CN-CLIP model: {cn_clip_vision_model_name} (Vision) from path (if provided): {clip_path}")
                    device_for_cn_clip = "cuda" if self.use_cuda else "cpu"
                    try:
                        self.clip_model_cn, _ = cn_clip_lib.load_from_name(
                            clip_path if os.path.isdir(clip_path) or os.path.isfile(
                                clip_path) else cn_clip_vision_model_name,
                            device=device_for_cn_clip,
                            download_root=None if (os.path.isdir(clip_path) or os.path.isfile(clip_path)) else (
                                clip_path if clip_path and not (os.path.isdir(clip_path) or os.path.isfile(
                                    clip_path)) else './pretrained_model/clip_cn/'))
                    except Exception as e_load:
                        logger.warning(
                            f"Failed to load CN-CLIP using '{clip_path}' as name/path, trying with default name and clip_path as download_root: {e_load}")
                        self.clip_model_cn, _ = cn_clip_lib.load_from_name(cn_clip_vision_model_name,
                                                                           device=device_for_cn_clip,
                                                                           download_root=clip_path if clip_path else './pretrained_model/clip_cn/')

                    for param in self.clip_model_cn.parameters(): param.requires_grad_(False)
                    logger.info("CN-CLIP model loaded successfully.")
                else:
                    logger.error("cn_clip library not imported. Cannot load CN-CLIP model for Weibo.")
            else:
                logger.warning(f"Unsupported dataset_type '{self.dataset_type}' for CLIP loading.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model for {self.dataset_type} from {clip_path}: {e}")

        feature_kernel_config_original = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        feature_kernel_config_direct320 = {1: 320}

        fk_to_use = feature_kernel_config_original if HAS_CUSTOM_LAYERS else feature_kernel_config_direct320

        if HAS_CUSTOM_LAYERS:
            output_dim_cnn_extractor = sum(fk_to_use.values()) if isinstance(fk_to_use, dict) else 320
        else:
            if isinstance(fk_to_use, dict) and len(fk_to_use) == 1 and list(fk_to_use.keys())[0] == 1:
                output_dim_cnn_extractor = list(fk_to_use.values())[0]
            else:
                output_dim_cnn_extractor = 320

        if output_dim_cnn_extractor != 320:
            logger.warning(
                f"cnn_extractor output dimension determined/defaulted to {output_dim_cnn_extractor}, "
                f"but 320 is generally expected. HAS_CUSTOM_LAYERS={HAS_CUSTOM_LAYERS}. fk_to_use={fk_to_use}."
            )

        expert_count = self.num_expert
        shared_count = expert_count * 2

        # Text Experts
        self.text_experts = nn.ModuleList()
        for _ in range(self.domain_num):
            if HAS_CUSTOM_LAYERS:
                experts = nn.ModuleList(
                    [cnn_extractor(self.text_dim, fk_to_use) for _ in range(expert_count)])  # Corrected: use fk_to_use
            else:
                experts = nn.ModuleList(
                    [cnn_extractor(self.text_dim, fk_to_use, out_features=output_dim_cnn_extractor) for _ in
                     range(expert_count)])
            self.text_experts.append(experts)

        # Image Experts
        self.image_experts = nn.ModuleList()
        for _ in range(self.domain_num):
            if HAS_CUSTOM_LAYERS:
                experts = nn.ModuleList(
                    [cnn_extractor(self.image_dim, fk_to_use) for _ in range(expert_count)])  # Corrected: use fk_to_use
            else:
                experts = nn.ModuleList(
                    [cnn_extractor(self.image_dim, fk_to_use, out_features=output_dim_cnn_extractor) for _ in
                     range(expert_count)])
            self.image_experts.append(experts)

        # Shared Experts - Text
        self.text_share_expert = nn.ModuleList()
        for _ in range(self.num_share):
            if HAS_CUSTOM_LAYERS:
                shared = nn.ModuleList(
                    [cnn_extractor(self.text_dim, fk_to_use) for _ in range(shared_count)])  # Corrected: use fk_to_use
            else:
                shared = nn.ModuleList(
                    [cnn_extractor(self.text_dim, fk_to_use, out_features=output_dim_cnn_extractor) for _ in
                     range(shared_count)])
            self.text_share_expert.append(shared)

        # Shared Experts - Image
        self.image_share_expert = nn.ModuleList()
        for _ in range(self.num_share):
            if HAS_CUSTOM_LAYERS:
                shared = nn.ModuleList(
                    [cnn_extractor(self.image_dim, fk_to_use) for _ in range(shared_count)])  # Corrected: use fk_to_use
            else:
                shared = nn.ModuleList(
                    [cnn_extractor(self.image_dim, fk_to_use, out_features=output_dim_cnn_extractor) for _ in
                     range(shared_count)])
            self.image_share_expert.append(shared)

        self.fusion_experts = nn.ModuleList()
        for _ in range(self.domain_num):
            experts = nn.ModuleList([nn.Sequential(nn.Linear(output_dim_cnn_extractor, output_dim_cnn_extractor),
                                                   nn.SiLU(),
                                                   nn.Linear(output_dim_cnn_extractor, output_dim_cnn_extractor)) for _
                                     in range(expert_count)])
            self.fusion_experts.append(experts)

        self.fusion_share_expert = nn.ModuleList()
        for _ in range(self.num_share):
            shared = nn.ModuleList([nn.Sequential(nn.Linear(output_dim_cnn_extractor, output_dim_cnn_extractor),
                                                  nn.SiLU(),
                                                  nn.Linear(output_dim_cnn_extractor, output_dim_cnn_extractor)) for _
                                    in range(shared_count)])
            self.fusion_share_expert.append(shared)

        gate_out_dim_specific_plus_shared = expert_count + shared_count
        fusion0_gate_out_dim = expert_count * 3

        self.text_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
                                                           nn.Linear(self.unified_dim,
                                                                     gate_out_dim_specific_plus_shared),
                                                           nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in
                                             range(self.domain_num)])
        self.image_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
                                                            nn.Linear(self.unified_dim,
                                                                      gate_out_dim_specific_plus_shared),
                                                            nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in
                                              range(self.domain_num)])
        self.fusion_gate_list0 = nn.ModuleList([nn.Sequential(
            nn.Linear(output_dim_cnn_extractor, output_dim_cnn_extractor // 2 if output_dim_cnn_extractor > 0 else 160),
            nn.SiLU(),
            nn.Linear(output_dim_cnn_extractor // 2 if output_dim_cnn_extractor > 0 else 160, fusion0_gate_out_dim),
            nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in range(self.domain_num)])

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)

        classifier_input_dim = output_dim_cnn_extractor
        self.text_classifier = MLP(classifier_input_dim, mlp_dims, dropout)
        self.image_classifier = MLP(classifier_input_dim, mlp_dims, dropout)
        self.fusion_classifier = MLP(classifier_input_dim, mlp_dims, dropout)
        self.max_classifier = MLP(classifier_input_dim, mlp_dims, dropout)

        h_dims_mlp_fusion = mlp_dims if mlp_dims else ([348] if HAS_CUSTOM_LAYERS else [classifier_input_dim])
        self.MLP_fusion = MLP_fusion(output_dim_cnn_extractor * 3, output_dim_cnn_extractor, h_dims_mlp_fusion, dropout)
        self.domain_fusion = MLP_fusion(output_dim_cnn_extractor, output_dim_cnn_extractor, h_dims_mlp_fusion, dropout)
        self.MLP_fusion0 = MLP_fusion(self.unified_dim * 2, self.unified_dim, h_dims_mlp_fusion, dropout)

        clip_embed_dim_hf = 512
        self.clip_fusion_input_dim = clip_embed_dim_hf * 2  # Default for HF CLIP
        if self.dataset_type in ['weibo', 'weibo21'] and self.clip_model_cn:
            try:
                img_proj_shape = getattr(self.clip_model_cn, 'vision_projection', None)
                txt_proj_shape = getattr(self.clip_model_cn, 'text_projection', None)
                if img_proj_shape is not None and hasattr(img_proj_shape, 'shape') and \
                        txt_proj_shape is not None and hasattr(txt_proj_shape, 'shape'):
                    self.clip_fusion_input_dim = img_proj_shape.shape[0] + txt_proj_shape.shape[0]
                    logger.info(f"CN-CLIP fusion input dim set to: {self.clip_fusion_input_dim}")
                else:
                    self.clip_fusion_input_dim = 1024
                    logger.warning(
                        f"Could not determine CN-CLIP projection shapes. Defaulting clip_fusion_input_dim to {self.clip_fusion_input_dim}")
            except Exception as e_clip_dim:
                logger.error(f"Error determining CN-CLIP fusion input dim: {e_clip_dim}. Defaulting to 1024.")
                self.clip_fusion_input_dim = 1024

        self.clip_fusion = MLP_fusion(self.clip_fusion_input_dim, output_dim_cnn_extractor, h_dims_mlp_fusion, dropout)

        self.att_mlp_text = MLP_fusion(output_dim_cnn_extractor, 2,
                                       [output_dim_cnn_extractor // 2 if output_dim_cnn_extractor > 0 else 174],
                                       dropout)
        self.att_mlp_img = MLP_fusion(output_dim_cnn_extractor, 2,
                                      [output_dim_cnn_extractor // 2 if output_dim_cnn_extractor > 0 else 174], dropout)
        self.att_mlp_mm = MLP_fusion(output_dim_cnn_extractor, 2,
                                     [output_dim_cnn_extractor // 2 if output_dim_cnn_extractor > 0 else 174], dropout)

        self.adaIN = AdaIN()
        self.mapping_IS_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                               nn.Linear(self.unified_dim // 2, 1))
        self.mapping_IS_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                                  nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                              nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                                 nn.Linear(self.unified_dim // 2, 1))

    def forward(self, **kwargs):
        inputs_ids = kwargs['content']
        masks = kwargs['content_masks']
        image_for_mae = kwargs['image']

        clip_image_input = kwargs['clip_image']
        clip_text_input_ids = kwargs['clip_text']
        clip_text_attention_mask = kwargs.get('clip_attention_mask', None)

        batch_size = inputs_ids.shape[0]
        device = inputs_ids.device

        cnn_out_dim = 320
        if self.text_experts and len(self.text_experts) > 0 and \
                self.text_experts[0] and len(self.text_experts[0]) > 0 and \
                hasattr(self.text_experts[0][0], 'out_dim'):
            cnn_out_dim = self.text_experts[0][0].out_dim
        elif not HAS_CUSTOM_LAYERS and hasattr(self, 'text_experts') and self.text_experts and self.text_experts[0] and \
                self.text_experts[0][0]:
            if hasattr(self.text_experts[0][0], 'out_dim'):
                cnn_out_dim = self.text_experts[0][0].out_dim

        text_feature_seq = None
        if self.bert:
            try:
                bert_outputs = self.bert(input_ids=inputs_ids, attention_mask=masks)
                text_feature_seq = bert_outputs.last_hidden_state
            except Exception as e:
                logger.error(f"Error during BERT forward pass: {e}")
        if text_feature_seq is None:
            text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.text_dim, device=device)

        image_feature_seq_mae = None
        if self.image_model_mae:
            try:
                image_feature_seq_mae = self.image_model_mae.forward_ying(image_for_mae)
            except Exception as e:
                logger.error(f"Error during MAE forward pass: {e}")
        if image_feature_seq_mae is None:
            image_feature_seq_mae = torch.zeros(batch_size, self.image_token_len_expected_mae, self.image_dim,
                                                device=device)

        clip_image_embed, clip_text_embed = None, None
        try:
            if self.dataset_type == 'gossipcop' and self.clip_model_hf:
                with torch.no_grad():
                    clip_img_out = self.clip_model_hf.get_image_features(pixel_values=clip_image_input)
                    clip_image_embed = clip_img_out / (clip_img_out.norm(dim=-1, keepdim=True) + 1e-9)
                    clip_txt_out = self.clip_model_hf.get_text_features(input_ids=clip_text_input_ids,
                                                                        attention_mask=clip_text_attention_mask)
                    clip_text_embed = clip_txt_out / (clip_txt_out.norm(dim=-1, keepdim=True) + 1e-9)
            elif self.dataset_type in ['weibo', 'weibo21'] and self.clip_model_cn:
                with torch.no_grad():
                    clip_image_embed = self.clip_model_cn.encode_image(clip_image_input.float())
                    clip_text_embed = self.clip_model_cn.encode_text(clip_text_input_ids)
                    if clip_image_embed is not None: clip_image_embed /= clip_image_embed.norm(dim=-1,
                                                                                               keepdim=True) + 1e-9
                    if clip_text_embed is not None: clip_text_embed /= clip_text_embed.norm(dim=-1, keepdim=True) + 1e-9
        except Exception as e:
            logger.error(f"Error during CLIP forward pass for {self.dataset_type}: {e}")

        actual_clip_fusion_in_dim = self.clip_fusion_input_dim

        expected_img_dim_clip = actual_clip_fusion_in_dim // 2
        expected_txt_dim_clip = actual_clip_fusion_in_dim - expected_img_dim_clip

        if clip_image_embed is None or clip_image_embed.shape[1] != expected_img_dim_clip:
            if clip_image_embed is not None: logger.debug(
                f"CLIP image embed dim mismatch for {self.dataset_type}: got {clip_image_embed.shape[1]}, expected {expected_img_dim_clip}. Using zeros.")
            clip_image_embed = torch.zeros(batch_size, expected_img_dim_clip, device=device)
        if clip_text_embed is None or clip_text_embed.shape[1] != expected_txt_dim_clip:
            if clip_text_embed is not None: logger.debug(
                f"CLIP text embed dim mismatch for {self.dataset_type}: got {clip_text_embed.shape[1]}, expected {expected_txt_dim_clip}. Using zeros.")
            clip_text_embed = torch.zeros(batch_size, expected_txt_dim_clip, device=device)

        clip_fusion_feature_combined = torch.cat((clip_image_embed, clip_text_embed), dim=-1).float()
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature_combined)

        text_atn_feature = self.text_attention(text_feature_seq, masks)
        image_atn_feature, _ = self.image_attention(image_feature_seq_mae)

        text_gate_input = text_atn_feature
        image_gate_input = image_atn_feature

        domain_idx = 0

        text_gate_out = self.text_gate_list[domain_idx](text_gate_input)
        image_gate_out = self.image_gate_list[domain_idx](image_gate_input)

        text_experts_feature_sum = torch.zeros(batch_size, cnn_out_dim, device=device)
        text_gate_share_expert_value_sum = torch.zeros(batch_size, cnn_out_dim, device=device)

        for j in range(self.num_expert):
            tmp_expert_feat = self.text_experts[domain_idx][j](text_feature_seq)
            text_experts_feature_sum += (tmp_expert_feat * text_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp_shared_feat = self.text_share_expert[0][j](text_feature_seq)
            gate_val = text_gate_out[:, self.num_expert + j].unsqueeze(1)
            text_experts_feature_sum += (tmp_shared_feat * gate_val)
            text_gate_share_expert_value_sum += (tmp_shared_feat * gate_val)

        att_text = F.softmax(self.att_mlp_text(text_experts_feature_sum), dim=-1)
        text_final_feat_0 = att_text[:, 0].unsqueeze(1) * text_experts_feature_sum

        image_experts_feature_sum = torch.zeros(batch_size, cnn_out_dim, device=device)
        image_gate_share_expert_value_sum = torch.zeros(batch_size, cnn_out_dim, device=device)

        for j in range(self.num_expert):
            tmp_expert_feat = self.image_experts[domain_idx][j](image_feature_seq_mae)
            image_experts_feature_sum += (tmp_expert_feat * image_gate_out[:, j].unsqueeze(1))
        for j in range(self.num_expert * 2):
            tmp_shared_feat = self.image_share_expert[0][j](image_feature_seq_mae)
            gate_val = image_gate_out[:, self.num_expert + j].unsqueeze(1)
            image_experts_feature_sum += (tmp_shared_feat * gate_val)
            image_gate_share_expert_value_sum += (tmp_shared_feat * gate_val)

        att_img = F.softmax(self.att_mlp_img(image_experts_feature_sum), dim=-1)
        image_final_feat_0 = att_img[:, 0].unsqueeze(1) * image_experts_feature_sum

        fusion_input_modalities = torch.cat(
            (clip_fusion_feature, text_gate_share_expert_value_sum, image_gate_share_expert_value_sum), dim=-1)
        fusion_shared_representation = self.MLP_fusion(fusion_input_modalities)

        fusion_gate_input0 = self.domain_fusion(fusion_shared_representation)
        fusion_gate_out0 = self.fusion_gate_list0[domain_idx](fusion_gate_input0)

        fusion_experts_feature_sum = torch.zeros(batch_size, cnn_out_dim, device=device)

        for n in range(self.num_expert):
            tmp_expert_feat = self.fusion_experts[domain_idx][n](fusion_shared_representation)
            fusion_experts_feature_sum += (tmp_expert_feat * fusion_gate_out0[:, n].unsqueeze(1))
        for n in range(self.num_expert * 2):
            tmp_shared_feat = self.fusion_share_expert[0][n](fusion_shared_representation)
            gate_val_idx = self.num_expert + n
            if gate_val_idx < fusion_gate_out0.shape[1]:
                fusion_experts_feature_sum += (tmp_shared_feat * fusion_gate_out0[:, gate_val_idx].unsqueeze(1))

        att_mm = F.softmax(self.att_mlp_mm(fusion_experts_feature_sum), dim=-1)
        fusion_final_feat_0 = att_mm[:, 0].unsqueeze(1) * fusion_experts_feature_sum

        text_logits = self.text_classifier(text_final_feat_0).squeeze(-1)
        image_logits = self.image_classifier(image_final_feat_0).squeeze(-1)
        fusion_logits = self.fusion_classifier(fusion_final_feat_0).squeeze(-1)

        all_modality_combined = text_final_feat_0 + image_final_feat_0 + fusion_final_feat_0
        final_logits = self.max_classifier(all_modality_combined).squeeze(-1)

        return final_logits, text_logits, image_logits, fusion_logits


class Trainer():
    def __init__(self,
                 emb_dim, mlp_dims,
                 bert_path, clip_path, mae_checkpoint_path,
                 dataset_type,
                 use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader,
                 category_dict, weight_decay, save_param_dir,
                 early_stop=10, epoches=100,
                 metric_key_for_early_stop='acc',
                 data_to_gpu_func=None
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.metric_key_for_early_stop = metric_key_for_early_stop
        self.save_param_dir = save_param_dir
        self.dataset_type = dataset_type
        self.data_to_gpu_func = data_to_gpu_func

        os.makedirs(self.save_param_dir, exist_ok=True)

        self.model = MultiDomainPLEFENDModel(
            emb_dim=self.emb_dim,
            mlp_dims=self.mlp_dims,
            bert_path=bert_path,
            clip_path=clip_path,
            mae_checkpoint_path=mae_checkpoint_path,
            dataset_type=self.dataset_type,
            dropout=self.dropout,
            use_cuda=self.use_cuda
        )

        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            logger.warning("CUDA not available/requested. Model on CPU.")

    def train(self):
        loss_fn = torch.nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        recorder = Recorder(self.early_stop, metric_key=self.metric_key_for_early_stop)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epoches} Training")
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                try:
                    if self.data_to_gpu_func:
                        batch_data = self.data_to_gpu_func(batch)
                    else:
                        batch_data = batch

                    if batch_data is None:
                        logger.warning(f"Skipping training batch {step_n} due to data loading/GPU transfer error.")
                        continue

                    label = batch_data.get('label')
                    if label is None:
                        logger.warning(f"Skipping training batch {step_n} due to missing label.")
                        continue

                    label = label.float()

                    final_logits, text_logits, image_logits, fusion_logits = self.model(**batch_data)

                    loss0 = loss_fn(final_logits, label)
                    loss1 = loss_fn(text_logits, label)
                    loss2 = loss_fn(image_logits, label)
                    loss3 = loss_fn(fusion_logits, label)

                    loss = loss0 + (loss1 + loss2 + loss3) / 3.0

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    avg_loss.add(loss.item())
                    train_data_iter.set_postfix(loss=avg_loss.item(), lr=optimizer.param_groups[0]['lr'])

                except Exception as e:
                    logger.exception(f"Error in training step {epoch + 1}-{step_n}: {e}")
                    if "CUDA out of memory" in str(e):
                        logger.error("CUDA out of memory. Try reducing batch size.")
                    continue

            if scheduler is not None:
                scheduler.step()

            logger.info(
                f'Training Epoch {epoch + 1} Done; Avg Loss: {avg_loss.item():.4f}; LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if self.val_loader is None:
                logger.warning("Validation loader not provided. Skipping validation.")
                continue

            try:
                val_results = self.test(self.val_loader)
                if not val_results:
                    logger.warning(f"Validation epoch {epoch + 1} returned no results. Skipping score processing.")
                    continue

                current_metric_val = val_results.get(self.metric_key_for_early_stop, 0.0)
                acc_val = val_results.get('acc', 0.0)
                f1_val = val_results.get('F1', 0.0)
                auc_val = val_results.get('auc', 0.0)

                log_msg = f"Validation E{epoch + 1}: Acc:{acc_val:.4f} F1:{f1_val:.4f} AUC:{auc_val:.4f} Tracked({self.metric_key_for_early_stop}):{current_metric_val:.4f}"
                if 'Real' in val_results and 'Fake' in val_results:
                    log_msg += f" | Real F1:{val_results['Real'].get('F1', 0.0):.4f}, Fake F1:{val_results['Fake'].get('F1', 0.0):.4f}"
                elif 'real' in val_results and 'fake' in val_results:
                    log_msg += f" | Real F1 (weibo):{val_results['real'].get('F1', 0.0):.4f}, Fake F1 (weibo):{val_results['fake'].get('F1', 0.0):.4f}"

                logger.info(log_msg)

                mark = recorder.add(val_results)
                if mark == 'save':
                    save_path = os.path.join(self.save_param_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Best model saved to {save_path} based on '{recorder.metric_key}'.")
                elif mark == 'esc':
                    logger.info(f"Early stopping triggered at epoch {epoch + 1} based on '{recorder.metric_key}'.")
                    break
            except Exception as e:
                logger.exception(f"Error during validation epoch {epoch + 1}: {e}")
                continue

        logger.info("Training loop finished.")
        recorder.showfinal()

        best_model_path = os.path.join(self.save_param_dir, 'best_model.pth')
        final_model_to_test_path = best_model_path
        loaded_best = False

        if os.path.exists(best_model_path):
            logger.info(f"Loading best model for final test: {best_model_path}")
            try:
                map_location = 'cpu' if not self.use_cuda else None
                self.model.load_state_dict(torch.load(best_model_path, map_location=map_location))
                if self.use_cuda: self.model.cuda()
                loaded_best = True
            except Exception as e:
                logger.error(
                    f"Failed to load best model state_dict from {best_model_path}: {e}. Using model from end of training.")
        else:
            logger.warning(f"Best model checkpoint {best_model_path} not found.")

        if not loaded_best:
            final_model_to_test_path = os.path.join(self.save_param_dir, 'final_training_state.pth')
            logger.warning(
                f"Best model not loaded. Testing with model state from end of training. Saving this state to: {final_model_to_test_path}")
            try:
                torch.save(self.model.state_dict(), final_model_to_test_path)
            except Exception as es:
                logger.error(f"Failed to save final training state model to {final_model_to_test_path}: {es}")

        final_results = None
        if self.test_loader is None:
            logger.warning("Test loader not provided. Skipping final test.")
            final_results = recorder.max if hasattr(recorder, 'max') and recorder.max else {
                "message": "No validation performed or no best model recorded."}
        else:
            logger.info(f"Starting final test with model: {final_model_to_test_path}")
            try:
                final_results = self.test(self.test_loader)
                if final_results:
                    acc = final_results.get('acc', 0.0);
                    f1 = final_results.get('F1', 0.0);
                    auc = final_results.get('auc', 0.0)
                    precision = final_results.get('precision', 0.0);
                    recall = final_results.get('recall', 0.0)
                    logger.info(
                        f"Final Test Results: Acc:{acc:.4f} F1:{f1:.4f} AUC:{auc:.4f} Precision:{precision:.4f} Recall:{recall:.4f}")

                    if 'Real' in final_results and 'Fake' in final_results:
                        real_m = final_results.get('Real', {})
                        fake_m = final_results.get('Fake', {})
                        log_final_class_summary = (
                            f"  GossipCop Style -> Real (label 1): P:{real_m.get('precision', 0.0):.4f} "
                            f"R:{real_m.get('recall', 0.0):.4f} "
                            f"F1:{real_m.get('F1', 0.0):.4f} | "
                            f"Fake (label 0): P:{fake_m.get('precision', 0.0):.4f} "
                            f"R:{fake_m.get('recall', 0.0):.4f} "
                            f"F1:{fake_m.get('F1', 0.0):.4f}"
                        )
                        logger.info(log_final_class_summary)
                    elif 'real' in final_results and 'fake' in final_results:
                        real_m_wb = final_results.get('real', {})
                        fake_m_wb = final_results.get('fake', {})
                        log_final_class_summary_wb = (
                            f"  Weibo Style -> Real (label 0): P:{real_m_wb.get('precision', 0.0):.4f} "
                            f"R:{real_m_wb.get('recall', 0.0):.4f} "
                            f"F1:{real_m_wb.get('F1', 0.0):.4f} | "
                            f"Fake (label 1): P:{fake_m_wb.get('precision', 0.0):.4f} "
                            f"R:{fake_m_wb.get('recall', 0.0):.4f} "
                            f"F1:{fake_m_wb.get('F1', 0.0):.4f}"
                        )
                        logger.info(log_final_class_summary_wb)
                else:
                    logger.error("Final test did not return valid results.")
            except Exception as e:
                logger.exception(f"Final test execution error: {e}")

        return final_results, final_model_to_test_path

    def test(self, dataloader):
        pred_probs_list, label_list, category_list = [], [], []
        if dataloader is None:
            logger.error("Test dataloader is None.");
            return {}

        self.model.eval()
        data_iter = tqdm.tqdm(dataloader, desc="Testing")

        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                try:
                    if self.data_to_gpu_func:
                        batch_data = self.data_to_gpu_func(batch)
                    else:
                        batch_data = batch

                    if batch_data is None:
                        logger.warning(f"Skipping test batch {step_n} due to data loading/GPU transfer error.")
                        continue

                    batch_label = batch_data.get('label')
                    batch_category = batch_data.get('category')

                    if batch_label is None:
                        logger.warning(f"Skipping test batch {step_n} due to missing label.")
                        continue

                    final_logits, _, _, _ = self.model(**batch_data)
                    batch_pred_probs = torch.sigmoid(final_logits)

                    label_list.extend(batch_label.cpu().numpy().tolist())
                    pred_probs_list.extend(batch_pred_probs.cpu().numpy().tolist())

                    if batch_category is not None:
                        category_list.extend(batch_category.cpu().numpy().tolist())
                    elif self.category_dict and isinstance(self.category_dict, dict) and len(self.category_dict) > 0:
                        category_list.extend([None] * batch_label.size(0))

                except Exception as e:
                    logger.exception(f"Error in testing batch {step_n}: {e}")
                    continue

        if not label_list or not pred_probs_list:
            logger.warning("No data successfully processed during test. Cannot calculate metrics.")
            return {}

        if self.category_dict and isinstance(self.category_dict, dict) and len(self.category_dict) > 0:
            if not category_list or len(category_list) != len(label_list):
                logger.warning(
                    f"Category list length mismatch or empty. Creating list of Nones for metrics. Labels: {len(label_list)}, Categories: {len(category_list if category_list else [])}")
                category_list = [None] * len(label_list)
        else:
            category_list = None

        metric_res = {}
        try:
            if self.dataset_type == 'gossipcop':
                metric_res = calculate_metrics(np.array(label_list), np.array(pred_probs_list),
                                               np.array(category_list) if category_list is not None else None,
                                               self.category_dict)
            elif self.dataset_type in ['weibo', 'weibo21']:
                cat_list_for_weibo = np.array(
                    category_list if category_list is not None else ([0] * len(label_list)))  # Default to 0 if None
                metric_res = metricsTrueFalse(np.array(label_list), np.array(pred_probs_list),
                                              cat_list_for_weibo, self.category_dict)
            else:
                logger.warning(
                    f"Unknown dataset type '{self.dataset_type}' for metrics calculation. Using default (calculate_metrics).")
                metric_res = calculate_metrics(np.array(label_list), np.array(pred_probs_list))

        except Exception as e:
            logger.exception(f"Error during metrics calculation for dataset {self.dataset_type}: {e}")
            metric_res = {}

        return metric_res