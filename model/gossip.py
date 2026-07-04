# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import logging

logger = logging.getLogger(__name__)

import os
import tqdm
import torch
from transformers import BertModel, CLIPModel
import torch.nn as nn
import torch.nn.functional as F

try:
    import mae
except ImportError:
    logger.error("Failed to import mae. Ensure mae.py is in the project root.")
    raise

try:
    from utils.utils_gossipcop import clipdata2gpu, Averager, calculate_metrics, Recorder
except ImportError as e:
    logger.warning(f"utils_gossipcop not found: {e}, falling back to utils.utils.")
    from utils.utils import clipdata2gpu, Averager, Recorder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None):
        import numpy as np
        preds = np.around(np.array(pred_probs)).astype(int)
        return {
            'acc': accuracy_score(label_list, preds),
            'metric': f1_score(label_list, preds, average='macro'),
            'precision': precision_score(label_list, preds, average='macro', zero_division=0),
            'recall': recall_score(label_list, preds, average='macro', zero_division=0),
        }

try:
    from .layers import *  # Assuming layers.py imports successfully
    from .pivot import *
except ImportError:
    logger.warning("Failed to import .layers or .pivot. Using placeholder definitions if needed.")

    class MaskAttention(nn.Module):
        def __init__(self, dim):
            super().__init__(); self.dim = dim

        def forward(self, feat, mask):
            if feat is None or mask is None: return torch.zeros(1, self.dim) if feat is None else torch.zeros(
                feat.shape[0], self.dim, device=feat.device)
            if mask.dim() == 2 and feat.dim() == 3: mask = mask.unsqueeze(-1)
            if mask.shape != feat.shape:
                if mask.shape[-1] == 1 and feat.shape[-1] == self.dim:
                    mask = mask.expand_as(feat)
                else:
                    logger.warning(
                        f"Mask shape {mask.shape} mismatch {feat.shape}. Using mean pooling."); return torch.mean(feat,
                                                                                                                  dim=1)
            masked_feat = feat * mask
            sum_masked_feat = torch.sum(masked_feat, dim=1);
            sum_mask = torch.sum(mask, dim=1)
            return sum_masked_feat / (sum_mask + 1e-9)

    class TokenAttention(nn.Module):
        def __init__(self, dim):
            super().__init__();
            self.dim = dim;
            self.attention_weights = nn.Linear(dim, 1)

        def forward(self, feat):
            if feat is None: return torch.zeros(1, self.dim), None
            if feat.dim() != 3 or feat.shape[2] != self.dim: logger.error(
                f"TokenAttention shape error. Expected (B, L, {self.dim}), got {feat.shape}"); return torch.zeros(
                feat.shape[0], self.dim, device=feat.device), None
            e = self.attention_weights(feat);
            alpha = torch.softmax(e, dim=1);
            context = torch.bmm(alpha.transpose(1, 2), feat).squeeze(1);
            return context, alpha

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
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

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

    class cnn_extractor(nn.Module):  # placeholder cnn_extractor
        def __init__(self, in_dim_seq, feature_kernel_unused, out_dim=320):
            super().__init__()
            self.out_dim = out_dim
            self.in_dim = in_dim_seq
            self.pool_and_reduce = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim),
                nn.ReLU()
            )

        def forward(self, x_seq):
            if x_seq is None: return torch.zeros(1, self.out_dim,
                                                 device=x_seq.device if hasattr(x_seq, 'device') else 'cpu')
            if x_seq.shape[-1] != self.in_dim:
                logger.warning(
                    f"Placeholder cnn_extractor input feature dim mismatch. Expected {self.in_dim}, got {x_seq.shape[-1]}. Returning zeros.")
                return torch.zeros(x_seq.shape[0], self.out_dim, device=x_seq.device)
            if x_seq.dim() == 3:
                pooled_x = torch.mean(x_seq, dim=1)
            elif x_seq.dim() == 2:
                pooled_x = x_seq
            else:
                logger.error(f"Placeholder cnn_extractor input shape error: {x_seq.shape}, returning zeros.");
                return torch.zeros(x_seq.shape[0] if x_seq.dim() > 0 else 1, self.out_dim, device=x_seq.device)
            output = self.pool_and_reduce(pooled_x)
            if output.shape[-1] != self.out_dim:
                logger.error(
                    f"Placeholder cnn_extractor output shape error. Expected (*, {self.out_dim}), got {output.shape}. Check Linear layer.")
                return torch.zeros(output.shape[0], self.out_dim, device=output.device)
            return output

    class LayerNorm(nn.Module):
        def __init__(self, dim, eps=1e-12): super().__init__(); self.norm = nn.LayerNorm(dim, eps=eps)

        def forward(self, x): return self.norm(x) if x is not None else None

    class TransformerLayer(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()

        def forward(self, x, mask=None): return self.fc(x)

    class MLP_trans(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = nn.Identity()

        def forward(self, x): return self.fc(x)

from timm.models.vision_transformer import Block

class SimpleGate(nn.Module):
    def __init__(self, dim=1): super(SimpleGate, self).__init__(); self.dim = dim

    def forward(self, x): x1, x2 = x.chunk(2, dim=self.dim); return x1 * x2

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

class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims,
                 bert_path_or_name,
                 clip_path_or_name,
                 out_channels, dropout, use_cuda=True,
                 text_token_len=197, image_token_len=197,
                 num_domains=2):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.use_cuda = use_cuda;
        self.num_expert = 6;
        self.domain_num = num_domains;  # dynamic from category_dict
        self.num_share = 1
        self.unified_dim = 768;
        self.text_dim = 768;
        self.image_dim = 768
        self.text_token_len_expected = text_token_len;
        self.image_token_len_expected = image_token_len + 1  # MAE adds CLS token
        self.bert_path = bert_path_or_name;
        self.clip_path = clip_path_or_name

        try:
            logger.info(f"Loading BERT: {self.bert_path}")
            self.bert = BertModel.from_pretrained(self.bert_path, local_files_only=True)
            logger.info("BERT loaded.")
            for p in self.bert.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.bert = self.bert.cuda()
        except Exception as e:
            logger.error(f"Failed BERT load {self.bert_path}: {e}")
            self.bert = None
        self.model_size = "base";
        mae_cp = f'./mae_pretrain_vit_{self.model_size}.pth'
        try:
            self.image_model = mae.__dict__[f"mae_vit_{self.model_size}_patch16"](norm_pix_loss=False)
            if os.path.exists(mae_cp):
                logger.info(f"Loading MAE weights: {mae_cp}")
                cp = torch.load(mae_cp, map_location='cpu')
                sd = cp['model'] if 'model' in cp else cp
                lr_msg = self.image_model.load_state_dict(sd, strict=False)
                logger.info(f"MAE load result: {lr_msg}")
            else:
                logger.warning(f"MAE checkpoint not found {mae_cp}. Random init.")
            for p in self.image_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.image_model = self.image_model.cuda()
        except Exception as e:
            logger.exception(f"Failed MAE load: {e}")
            self.image_model = None
        try:
            logger.info(f"Loading CLIP: {self.clip_path}")
            self.clip_model = CLIPModel.from_pretrained(self.clip_path, local_files_only=True)
            logger.info("CLIP loaded.")
            for p in self.clip_model.parameters(): p.requires_grad_(False)
            if self.use_cuda: self.clip_model = self.clip_model.cuda()
        except Exception as e:
            logger.error(f"Failed CLIP load {self.clip_path}: {e}")
            self.clip_model = None

        fk = {1: 320}  # e.g. use one conv with kernel_size=1, output 320 features

        expert_count = self.num_expert  # 6
        shared_count = expert_count * 2  # 12

        self.text_experts = nn.ModuleList(
            [nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(expert_count)]) for _ in
             range(self.domain_num)])
        self.image_experts = nn.ModuleList(
            [nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(expert_count)]) for _ in
             range(self.domain_num)])  # Assuming image_experts uses the same cnn_extractor
        self.fusion_experts = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(expert_count)]) for _ in
                                             range(self.domain_num)])
        self.text_share_expert = nn.ModuleList(
            [nn.ModuleList([cnn_extractor(self.text_dim, fk) for _ in range(shared_count)]) for _ in
             range(self.num_share)])
        self.image_share_expert = nn.ModuleList(
            [nn.ModuleList([cnn_extractor(self.image_dim, fk) for _ in range(shared_count)]) for _ in
             range(self.num_share)])
        self.fusion_share_expert = nn.ModuleList([nn.ModuleList(
            [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(shared_count)]) for _ in
                                                  range(self.num_share)])

        gate_out_dim = expert_count + shared_count  # 6 + 12 = 18
        fusion0_out_dim = expert_count * 3  # 6 * 3 = 18

        self.image_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
                                                            nn.Linear(self.unified_dim, gate_out_dim),
                                                            nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in
                                              range(self.domain_num)])
        self.text_gate_list = nn.ModuleList([nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim), nn.SiLU(),
                                                           nn.Linear(self.unified_dim, gate_out_dim),
                                                           nn.Dropout(dropout), nn.Softmax(dim=1)) for _ in
                                             range(self.domain_num)])
        self.fusion_gate_list0 = nn.ModuleList([nn.Sequential(nn.Linear(320, 160), nn.SiLU(),
                                                              nn.Linear(160, fusion0_out_dim), nn.Dropout(dropout),
                                                              nn.Softmax(dim=1)) for _ in range(self.domain_num)])

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)

        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        self.max_classifier = MLP(320, mlp_dims, dropout)

        h_dims = mlp_dims if mlp_dims else [348]  # Default hidden dims for MLP_fusion if not provided
        self.MLP_fusion = MLP_fusion(960, 320, h_dims, dropout)
        self.domain_fusion = MLP_fusion(320, 320, h_dims, dropout)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, h_dims, dropout)
        self.clip_fusion = MLP_fusion(1024, 320, h_dims, dropout)

        self.att_mlp_text = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_img = MLP_fusion(320, 2, [174], dropout)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], dropout)

        self.mapping_IS_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                               nn.Linear(self.unified_dim // 2, 1))
        self.mapping_IS_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                                  nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_mu = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                              nn.Linear(self.unified_dim // 2, 1))
        self.mapping_T_MLP_sigma = nn.Sequential(nn.Linear(1, self.unified_dim // 2), nn.SiLU(),
                                                 nn.Linear(self.unified_dim // 2, 1))
        self.adaIN = AdaIN()

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        image_for_mae = kwargs['image']
        clip_pixel_values = kwargs['clip_image']
        clip_input_ids = kwargs['clip_text']
        clip_attention_mask = kwargs.get('clip_attention_mask', None)
        batch_size = inputs.shape[0]
        device = inputs.device

        text_feature_seq, image_feature_seq, clip_image_embed, clip_text_embed = None, None, None, None

        if self.bert:
            try:
                bert_outputs = self.bert(input_ids=inputs, attention_mask=masks)
                text_feature_seq = bert_outputs.last_hidden_state
            except Exception as e:
                logger.error(f"BERT error: {e}")
                text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.unified_dim,
                                               device=device)
        else:
            text_feature_seq = torch.zeros(batch_size, self.text_token_len_expected, self.unified_dim, device=device)

        if self.image_model:
            try:
                image_feature_seq = self.image_model.forward_ying(image_for_mae)
            except Exception as e:
                logger.error(f"MAE error: {e}")
                image_feature_seq = torch.zeros(batch_size, self.image_token_len_expected, self.unified_dim,
                                                device=device)
        else:
            image_feature_seq = torch.zeros(batch_size, self.image_token_len_expected, self.unified_dim, device=device)

        clip_output_dim = 512
        if self.clip_model:
            try:
                with torch.no_grad():
                    clip_img_out = self.clip_model.get_image_features(pixel_values=clip_pixel_values)
                    clip_image_embed = clip_img_out / (clip_img_out.norm(dim=-1, keepdim=True) + 1e-8)
                    clip_txt_out = self.clip_model.get_text_features(input_ids=clip_input_ids,
                                                                     attention_mask=clip_attention_mask)
                    clip_text_embed = clip_txt_out / (clip_txt_out.norm(dim=-1, keepdim=True) + 1e-8)
            except Exception as e:
                logger.error(f"CLIP error: {e}")
        if clip_image_embed is None: clip_image_embed = torch.zeros(batch_size, clip_output_dim, device=device)
        if clip_text_embed is None: clip_text_embed = torch.zeros(batch_size, clip_output_dim, device=device)

        text_atn_feature = self.text_attention(text_feature_seq, masks)
        image_atn_feature, _ = self.image_attention(image_feature_seq)
        clip_fusion_feature_in = torch.cat((clip_image_embed, clip_text_embed), dim=-1).float()
        clip_fusion_feature = self.clip_fusion(clip_fusion_feature_in)
        domain_idx = 0
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        output_shape_text = (batch_size, 320);
        text_experts_feature_sum = torch.zeros(output_shape_text, device=device);
        text_gate_share_expert_value_sum = torch.zeros(output_shape_text, device=device)
        for j in range(self.num_expert):
            tmp = self.text_experts[domain_idx][j](text_feature_seq)  # Expected shape (B, 320) with corrected fk
            gate_val_for_expert = text_gate_out[:, j].unsqueeze(1)  # Expected shape (B, 1)
            text_experts_feature_sum += (tmp * gate_val_for_expert)
        for j in range(self.num_expert * 2):
            tmp = self.text_share_expert[0][j](text_feature_seq)
            gate_val_for_shared_expert = text_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            text_experts_feature_sum += (tmp * gate_val_for_shared_expert)
            text_gate_share_expert_value_sum += (tmp * gate_val_for_shared_expert)
        att_text = F.softmax(self.att_mlp_text(text_experts_feature_sum), dim=-1)
        text_gate_expert_value = [att_text[:, i].unsqueeze(1) * text_experts_feature_sum for i in range(2)]

        output_shape_image = (batch_size, 320);
        image_experts_feature_sum = torch.zeros(output_shape_image, device=device);
        image_gate_share_expert_value_sum = torch.zeros(output_shape_image, device=device)
        for j in range(self.num_expert):
            tmp = self.image_experts[domain_idx][j](image_feature_seq)
            gate_val_for_expert = image_gate_out[:, j].unsqueeze(1)
            image_experts_feature_sum += (tmp * gate_val_for_expert)
        for j in range(self.num_expert * 2):
            tmp = self.image_share_expert[0][j](image_feature_seq)
            gate_val_for_shared_expert = image_gate_out[:, (self.num_expert + j)].unsqueeze(1)
            image_experts_feature_sum += (tmp * gate_val_for_shared_expert)
            image_gate_share_expert_value_sum += (tmp * gate_val_for_shared_expert)
        att_img = F.softmax(self.att_mlp_img(image_experts_feature_sum), dim=-1)
        image_gate_expert_value = [att_img[:, i].unsqueeze(1) * image_experts_feature_sum for i in range(2)]

        text_for_fusion = text_gate_share_expert_value_sum;
        image_for_fusion = image_gate_share_expert_value_sum
        fusion_share_feature_in = torch.cat((clip_fusion_feature, text_for_fusion, image_for_fusion), dim=-1);
        fusion_share_feature = self.MLP_fusion(fusion_share_feature_in)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature);
        fusion_gate_out0 = self.fusion_gate_list0[domain_idx](fusion_gate_input0)
        output_shape_fusion = (batch_size, 320);
        fusion_experts_feature_sum = torch.zeros(output_shape_fusion, device=device)
        for n in range(self.num_expert):
            tmp = self.fusion_experts[domain_idx][n](fusion_share_feature)
            gate_val = fusion_gate_out0[:, n].unsqueeze(1)
            fusion_experts_feature_sum += (tmp * gate_val)
        for n in range(self.num_expert * 2):
            tmp = self.fusion_share_expert[0][n](fusion_share_feature)
            gate_val = fusion_gate_out0[:, self.num_expert + n].unsqueeze(1)
            fusion_experts_feature_sum += (tmp * gate_val)
        att_mm = F.softmax(self.att_mlp_mm(fusion_experts_feature_sum), dim=-1)
        fusion_gate_expert_value0 = [att_mm[:, i].unsqueeze(1) * fusion_experts_feature_sum for i in range(2)]

        text_final_features = text_gate_expert_value[0];
        image_final_features = image_gate_expert_value[0];
        fusion_final_features = fusion_gate_expert_value0[0]
        text_logits = self.text_classifier(text_final_features).squeeze(-1);
        image_logits = self.image_classifier(image_final_features).squeeze(-1);
        fusion_logits = self.fusion_classifier(fusion_final_features).squeeze(-1)
        all_modality = text_final_features + image_final_features + fusion_final_features;
        final_logits = self.max_classifier(all_modality).squeeze(-1)
        return final_logits, text_logits, image_logits, fusion_logits

class Trainer():
    def __init__(self, emb_dim, mlp_dims,
                 bert_path_or_name, clip_path_or_name,
                 use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict, weight_decay,
                 save_param_dir, early_stop=10, epoches=100,
                 metric_key_for_early_stop='acc'):  # <--- change default to 'acc'
        self.lr = lr;
        self.weight_decay = weight_decay;
        self.train_loader = train_loader
        self.test_loader = test_loader;
        self.val_loader = val_loader;
        self.early_stop = early_stop
        self.epoches = epoches;
        self.category_dict = category_dict;
        self.use_cuda = use_cuda
        self.emb_dim = emb_dim;
        self.mlp_dims = mlp_dims;
        self.dropout = dropout
        self.metric_key_for_early_stop = metric_key_for_early_stop
        self.save_param_dir = save_param_dir
        os.makedirs(self.save_param_dir, exist_ok=True)

        num_domains = len(category_dict) if category_dict else 2
        self.model = MultiDomainPLEFENDModel(
            emb_dim=self.emb_dim, mlp_dims=self.mlp_dims,
            bert_path_or_name=bert_path_or_name,
            clip_path_or_name=clip_path_or_name,
            out_channels=320,  # This parameter may not be directly used in the current model
            dropout=self.dropout, use_cuda=self.use_cuda,
            num_domains=num_domains
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
            self.model.train();
            train_data_iter = tqdm.tqdm(self.train_loader);
            avg_loss = Averager()
            for step_n, batch in enumerate(train_data_iter):
                try:
                    batch_data = clipdata2gpu(batch)
                    if batch_data is None:
                        logger.warning(f"Skipping batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    label = batch_data.get('label')
                    if label is None:
                        logger.warning(f"Skipping batch {step_n} due to missing label.")
                        continue

                    final_logits, text_logits, image_logits, fusion_logits = self.model(**batch_data)

                    loss0 = loss_fn(final_logits, label.float())
                    loss1 = loss_fn(text_logits, label.float())
                    loss2 = loss_fn(image_logits, label.float())
                    loss3 = loss_fn(fusion_logits, label.float())
                    loss = loss0 + (loss1 + loss2 + loss3) / 3.0
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step()
                    avg_loss.add(loss.item())
                    train_data_iter.set_description(f"Epoch {epoch + 1}/{self.epoches}")
                    train_data_iter.set_postfix(loss=avg_loss.item(), lr=optimizer.param_groups[0]['lr'])
                except Exception as e:
                    if "size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                        logger.error(f"Tensor mismatch error at Train step {epoch}-{step_n}: {e}", exc_info=True)
                    elif "collate_fn" in str(e) or "image" in str(e).lower() or "channel" in str(e).lower():
                        logger.error(f"Image processing related error at Train step {epoch}-{step_n}: {e}",
                                     exc_info=True)
                    else:
                        logger.exception(f"Train step {step_n} error: {e}")
                    continue

            if scheduler is not None: scheduler.step()
            logger.info(
                f'Train Epoch {epoch + 1} Done; Avg Loss: {avg_loss.item():.4f}; LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if self.val_loader is None:
                logger.warning("Val loader not provided, skipping validation.")
                continue  # Skip validation

            try:
                val_results = self.test(self.val_loader)
                if not val_results:  # Check if val_results is empty or invalid
                    logger.warning(
                        f"Val epoch {epoch + 1} did not return valid results. Skipping score processing for this epoch.")
                    continue

                current_metric_val = val_results.get(self.metric_key_for_early_stop, 0.0)
                acc_val = val_results.get('acc', 0.0)
                f1_val = val_results.get('F1', 0.0)
                auc_val = val_results.get('auc', 0.0)

                logger.info(
                    f"Val E{epoch + 1}: Acc:{acc_val:.4f} F1:{f1_val:.4f} AUC:{auc_val:.4f} Tracked({self.metric_key_for_early_stop}):{current_metric_val:.4f}")
                logger.info(f"  Real:{repr(val_results.get('real', {}))}, Fake:{repr(val_results.get('fake', {}))}")

                mark = recorder.add(val_results)  # recorder.add should handle empty val_results if designed to
                if mark == 'save':
                    save_p = os.path.join(self.save_param_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_p)
                    logger.info(f"Best model saved based on '{self.metric_key_for_early_stop}': {save_p}")
                elif mark == 'esc':
                    logger.info(f"Early stopping triggered based on '{self.metric_key_for_early_stop}'.")
                    break
            except Exception as e:
                logger.exception(f"Val epoch {epoch + 1} error: {e}")
                continue

        logger.info("Training loop finished.");
        recorder.showfinal()
        best_model_path = os.path.join(self.save_param_dir, 'best_model.pth');
        loaded_best = False
        final_model_to_test_path = best_model_path
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model for final test: {best_model_path}")
            try:
                self.model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
                loaded_best = True
            except Exception as e:
                logger.error(f"Failed to load best model state_dict: {e}. Using model state from end of training.")

        if not loaded_best:  # If best model failed to load
            final_model_to_test_path = os.path.join(self.save_param_dir, 'final_training_state.pth')
            logger.warning(
                f"Best model not loaded. Testing with model state from end of training. Saving this state to: {final_model_to_test_path}")
            try:
                torch.save(self.model.state_dict(), final_model_to_test_path)
            except Exception as es:
                logger.error(f"Failed to save final training state model: {es}")

        final_results = None
        if self.test_loader is None:
            logger.warning("Test loader not provided. Skipping final test.")
            final_results = recorder.max if hasattr(recorder, 'max') else None
        else:
            logger.info("Starting final test with the chosen model...");
            try:
                final_results = self.test(self.test_loader)  # Call test, returns dict with detailed metrics
                if final_results:
                    acc = final_results.get('acc', 0.0);
                    f1 = final_results.get('F1', 0.0);
                    auc = final_results.get('auc', 0.0)
                    precision = final_results.get('precision', 0.0);
                    recall = final_results.get('recall', 0.0)
                    logger.info(
                        f"Final Test Results: Acc:{acc:.4f} F1:{f1:.4f} AUC:{auc:.4f} Precision:{precision:.4f} Recall:{recall:.4f}")

                    real_m = final_results.get('Real', {})  # Get 'Real' sub-dict from results
                    fake_m = final_results.get('Fake', {})  # Get 'Fake' sub-dict from results

                    log_final_class_summary = (
                        f"  Real (label=1): P:{real_m.get('precision', 0.0):.4f} "
                        f"R:{real_m.get('recall', 0.0):.4f} "
                        f"F1:{real_m.get('F1', 0.0):.4f} | "
                        f"Fake (label=0): P:{fake_m.get('precision', 0.0):.4f} "
                        f"R:{fake_m.get('recall', 0.0):.4f} "
                        f"F1:{fake_m.get('F1', 0.0):.4f}"
                    )
                    logger.info(log_final_class_summary)  # Print detailed class metrics

                else:
                    logger.error("Final test did not return valid results.")
            except Exception as e:
                logger.exception(f"Final test execution error: {e}")
        return final_results, final_model_to_test_path

    def test(self, dataloader):
        pred_probs, label_list, category_list = [], [], []
        if dataloader is None:
            logger.error("Test dataloader is None.");
            return {}
        self.model.eval();
        data_iter = tqdm.tqdm(dataloader, desc="Testing")
        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                try:
                    batch_data = clipdata2gpu(batch)
                    if batch_data is None:
                        logger.warning(f"Skipping test batch {step_n} due to data loading/GPU transfer error.")
                        continue
                    batch_label = batch_data.get('label')
                    batch_category = batch_data.get('category')  # used in calculate_metrics
                    if batch_label is None:  # category is not strictly needed, but label is
                        logger.warning(f"Skipping test batch {step_n} due to missing label.")
                        continue
                    if batch_category is None:  # If category_dict is nonempty, category should also be available
                        logger.debug(
                            f"Test batch {step_n} missing category, will pass None to metrics if category_dict used.")

                    final_logits, _, _, _ = self.model(**batch_data)
                    batch_pred_prob = torch.sigmoid(final_logits)
                    label_list.extend(batch_label.cpu().numpy().tolist())
                    pred_probs.extend(batch_pred_prob.cpu().numpy().tolist())
                    if batch_category is not None:
                        category_list.extend(batch_category.cpu().numpy().tolist())
                    else:  # If category missing but calculate_metrics needs it, fill placeholder
                        category_list.extend([None] * batch_label.size(0))

                except Exception as e:
                    logger.exception(f"Test batch {step_n} error: {e}")
                    continue  # Continue to next batch

        if not label_list or not pred_probs:  # Ensure there is data to compute metrics
            logger.warning("No data was successfully processed in test function to calculate metrics.")
            return {}

        if len(category_list) != len(label_list) and self.category_dict:
            logger.warning(
                f"Mismatch in length of category_list ({len(category_list)}) and label_list ({len(label_list)}). Filling with None.")
            category_list = (category_list + [None] * len(label_list))[:len(label_list)]

        try:
            if self.category_dict:
                metric_res = calculate_metrics(label_list, pred_probs, category_list, self.category_dict)
            else:
                metric_res = calculate_metrics(label_list,
                                               pred_probs)  # Assuming calculate_metrics can work without category_list and category_dict
        except Exception as e:
            logger.exception(f"Metrics calculation error: {e}");
            metric_res = {}
        return metric_res
# Author: 
