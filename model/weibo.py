# -*-codeing = utf-8 -*-
# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import mae
import torch.nn.functional as F
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

from utils.utils_weibo import clipdata2gpu, Averager, metricsTrueFalse, Recorder
from .layers import MLP, MaskAttention, TokenAttention, cnn_extractor, MLP_fusion, clip_fuion
from timm.models.vision_transformer import Block

try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name
except ImportError:
    print("Warning: cn_clip library not found. CLIP functionalities will not work.")
    clip = None
    load_from_name = None

class MultiDomainPLEFENDModel(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout,
                 reasoning_emb_dim=768, num_manipulation_classes=0):
        super(MultiDomainPLEFENDModel, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768

        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}

        text_expert_list = [
            nn.ModuleList([cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert)]) for _ in
            range(self.domain_num)]
        self.text_experts = nn.ModuleList(text_expert_list)
        image_expert_list = [
            nn.ModuleList([cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)]) for _ in
            range(self.domain_num)]
        self.image_experts = nn.ModuleList(image_expert_list)
        fusion_expert_list = [nn.ModuleList(
            [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in range(self.num_expert)]) for _
                              in range(self.domain_num)]
        self.fusion_experts = nn.ModuleList(fusion_expert_list)
        text_share = [cnn_extractor(self.text_dim, feature_kernel) for _ in range(self.num_expert * 2)]
        image_share = [cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert * 2)]
        fusion_share = [nn.Sequential(nn.Linear(320, 320), nn.SiLU(), nn.Linear(320, 320)) for _ in
                        range(self.num_expert * 2)]
        self.text_share_expert = nn.ModuleList([nn.ModuleList(text_share)])
        self.image_share_expert = nn.ModuleList([nn.ModuleList(image_share)])
        self.fusion_share_expert = nn.ModuleList([nn.ModuleList(fusion_share)])
        gate_output_dim = self.num_expert * 3
        self.text_gate_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.text_dim, gate_output_dim), nn.Softmax(dim=1)) for _ in
             range(self.domain_num)])
        self.image_gate_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.image_dim, gate_output_dim), nn.Softmax(dim=1)) for _ in
             range(self.domain_num)])
        self.fusion_gate_list0 = nn.ModuleList(
            [nn.Sequential(nn.Linear(320, gate_output_dim), nn.Softmax(dim=1)) for _ in range(self.domain_num)])

        self.text_attention = MaskAttention(self.text_dim)
        self.image_attention = TokenAttention(self.image_dim)

        feature_dim_after_experts = 320
        self.text_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.image_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.fusion_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)
        self.max_classifier = MLP(feature_dim_after_experts, mlp_dims, dropout)

        self.MLP_fusion = MLP_fusion(320 * 3, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        if clip is not None:
            self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        else:
            self.clip_fusion = None

        self.att_mlp_text = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(feature_dim_after_experts, 2, [174], 0.1)

        self.model_size = "base"
        try:
            self.image_model = mae.__dict__[f"mae_vit_{self.model_size}_patch16"](norm_pix_loss=False)
            checkpoint = torch.load(f'./mae_pretrain_vit_{self.model_size}.pth', map_location='cpu')
            self.image_model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available(): self.image_model.cuda()
            for param in self.image_model.parameters(): param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load MAE model. Error: {e}")
            self.image_model = None

        if clip is not None:
            try:
                clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.ClipModel, _ = load_from_name("ViT-B-16", device=clip_device, download_root='./')
            except Exception as e:
                print(f"Warning: Could not load CLIP model. Error: {e}")
                self.ClipModel = None
        else:
            self.ClipModel = None

        refinement_input_dim = feature_dim_after_experts * 3  # Concatenation of F_text, F_image, F_cross
        self.refinement_module_text = MLP_fusion(refinement_input_dim, feature_dim_after_experts,
                                                 [refinement_input_dim // 2], dropout)
        self.refinement_module_image = MLP_fusion(refinement_input_dim, feature_dim_after_experts,
                                                  [refinement_input_dim // 2], dropout)
        self.refinement_module_cross = MLP_fusion(refinement_input_dim, feature_dim_after_experts,
                                                  [refinement_input_dim // 2], dropout)

        self.num_manipulation_classes = num_manipulation_classes
        if self.num_manipulation_classes > 0:
            self.manipulation_classifier = MLP(feature_dim_after_experts, [self.num_manipulation_classes], dropout)
        else:
            self.manipulation_classifier = None

    def get_expert_output(self, features, gate_outputs, specific_experts, shared_experts):
        num_specific_experts = len(specific_experts)
        shared_expert_outputs = [expert(features) for expert in shared_experts[0]]
        specific_expert_outputs = [expert(features) for expert in specific_experts]

        all_experts_output = shared_expert_outputs + specific_expert_outputs

        expert_outputs_sum = torch.sum(torch.stack(all_experts_output, dim=1) * gate_outputs.unsqueeze(-1), dim=1)

        shared_gate_outputs = gate_outputs[:, :len(shared_expert_outputs)]
        shared_expert_outputs_sum = torch.sum(
            torch.stack(shared_expert_outputs, dim=1) * shared_gate_outputs.unsqueeze(-1), dim=1)

        return expert_outputs_sum, shared_expert_outputs_sum

    def forward(self, **kwargs):
        self.ClipModel = self.ClipModel.float()  # Force CLIP into single-precision mode
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        image_raw = kwargs['image']
        text_feature_full = self.bert(inputs, attention_mask=masks)[0]
        if self.image_model:
            try:
                image_feature_full = self.image_model.forward_ying(image_raw)
            except AttributeError:
                image_feature_full = self.image_model.forward_features(image_raw)
        else:
            image_feature_full = torch.zeros_like(text_feature_full)

        clip_image_input, clip_text_input = kwargs.get('clip_image'), kwargs.get('clip_text')
        clip_fusion_feature = torch.zeros(text_feature_full.size(0), 320, device=text_feature_full.device)
        if self.ClipModel and self.clip_fusion and clip_image_input is not None and clip_text_input is not None:
            with torch.no_grad():
                img_feat = self.ClipModel.encode_image(clip_image_input)
                txt_feat = self.ClipModel.encode_text(clip_text_input.long())
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            clip_concat_feature = torch.cat((img_feat, txt_feat), dim=-1).float()
            clip_fusion_feature = self.clip_fusion(clip_concat_feature)

        text_atn_feature = self.text_attention(text_feature_full, masks)
        image_atn_feature, _ = self.image_attention(image_feature_full)
        domain_idx = kwargs.get('domain_idx', 0) % self.domain_num
        text_gate_out = self.text_gate_list[domain_idx](text_atn_feature)
        image_gate_out = self.image_gate_list[domain_idx](image_atn_feature)

        text_experts_output, text_shared_output = self.get_expert_output(text_feature_full, text_gate_out,
                                                                         self.text_experts[domain_idx],
                                                                         self.text_share_expert)
        F_text = text_experts_output * F.softmax(self.att_mlp_text(text_experts_output), dim=-1)[:, 0].unsqueeze(1)

        image_experts_output, image_shared_output = self.get_expert_output(image_feature_full, image_gate_out,
                                                                           self.image_experts[domain_idx],
                                                                           self.image_share_expert)
        F_image = image_experts_output * F.softmax(self.att_mlp_img(image_experts_output), dim=-1)[:, 0].unsqueeze(1)

        fusion_base = self.MLP_fusion(torch.cat((clip_fusion_feature, text_shared_output, image_shared_output), dim=-1))
        fusion_gate_out = self.fusion_gate_list0[domain_idx](self.domain_fusion(fusion_base))
        fusion_experts_output, _ = self.get_expert_output(fusion_base, fusion_gate_out, self.fusion_experts[domain_idx],
                                                          self.fusion_share_expert)
        F_cross = fusion_experts_output * F.softmax(self.att_mlp_mm(fusion_experts_output), dim=-1)[:, 0].unsqueeze(1)

        F_concat_all_views = torch.cat((F_text, F_image, F_cross), dim=1)
        P_delta_text = self.refinement_module_text(F_concat_all_views)
        P_delta_image = self.refinement_module_image(F_concat_all_views)
        P_delta_cross = self.refinement_module_cross(F_concat_all_views)

        F_prime_text = F_text + P_delta_text
        F_prime_image = F_image + P_delta_image
        F_prime_cross = F_cross + P_delta_cross

        if self.training:
            text_drop_mask = (torch.rand(F_prime_text.size(0), 1, device=F_prime_text.device) > 0.1).float() / 0.9
            F_prime_text = F_prime_text * text_drop_mask

            image_drop_mask = (torch.rand(F_prime_image.size(0), 1, device=F_prime_image.device) > 0.1).float() / 0.9
            F_prime_image = F_prime_image * image_drop_mask

        text_logits = self.text_classifier(F_prime_text).squeeze(1)
        image_logits = self.image_classifier(F_prime_image).squeeze(1)
        fusion_logits = self.fusion_classifier(F_prime_cross).squeeze(1)
        text_prob = torch.sigmoid(text_logits)
        image_prob = torch.sigmoid(image_logits)
        fusion_prob = torch.sigmoid(fusion_logits)

        all_modality_combined_prime = F_prime_text + F_prime_image + F_prime_cross
        final_logits = self.max_classifier(all_modality_combined_prime).squeeze(1)
        final_prob = torch.sigmoid(final_logits)

        manipulation_pred_logits = self.manipulation_classifier(
            all_modality_combined_prime) if self.manipulation_classifier else None

        return (final_prob, text_prob, image_prob, fusion_prob,
                F_text, F_image, F_cross,
                P_delta_text, P_delta_image, P_delta_cross,
                manipulation_pred_logits)

class Trainer():
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout,
                 train_loader, val_loader, test_loader, category_dict,
                 weight_decay, save_param_dir, reasoning_emb_dim=768,
                 num_manipulation_classes=0, lambda_reasoning_align=0.1,
                 lambda_manipulation_predict=0, early_stop=100, epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.early_stop, self.epoches = early_stop, epoches
        self.category_dict, self.use_cuda = category_dict, use_cuda
        self.emb_dim, self.mlp_dims, self.bert, self.dropout = emb_dim, mlp_dims, bert, dropout
        self.save_param_dir = save_param_dir
        if not os.path.exists(self.save_param_dir): os.makedirs(self.save_param_dir, exist_ok=True)

        self.lambda_reasoning_align = lambda_reasoning_align
        self.lambda_manipulation_predict = lambda_manipulation_predict

        self.model = MultiDomainPLEFENDModel(
            emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
            out_channels=320, dropout=self.dropout, reasoning_emb_dim=reasoning_emb_dim,
            num_manipulation_classes=num_manipulation_classes)
        if self.use_cuda: self.model = self.model.cuda()

        self.bce_loss = BinaryFocalLoss(gamma=2.0, alpha=0.75)
        self.mse_loss = nn.MSELoss()
        if num_manipulation_classes > 0: self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.98)
        self.model_save_filename = 'parameter_calibration_distill.pkl'

    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoches):
            self.model.train()
            train_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            for batch in train_iter:
                batch_data = clipdata2gpu(batch, self.use_cuda)
                labels = batch_data['label'].float()

                (final_prob, text_prob, image_prob, fusion_prob,
                 F_text, F_image, F_cross,
                 P_delta_text, P_delta_image, P_delta_cross,
                 manip_logits) = self.model(**batch_data)

                loss_main = self.bce_loss(final_prob, labels)
                loss_aux = (self.bce_loss(text_prob, labels) + self.bce_loss(image_prob, labels) + self.bce_loss(
                    fusion_prob, labels)) / 3.0
                loss_detection = loss_main + loss_aux

                teacher_E_text = batch_data.get('teacher_reasoning_text_emb')
                teacher_E_image = batch_data.get('teacher_reasoning_image_emb')
                teacher_E_cross = batch_data.get('teacher_reasoning_cross_emb')
                loss_distill = torch.tensor(0.0, device=labels.device)

                if all(e is not None for e in [teacher_E_text, teacher_E_image, teacher_E_cross]):
                    Delta_text = teacher_E_text - F_text
                    Delta_image = teacher_E_image - F_image
                    Delta_cross = teacher_E_cross - F_cross

                    loss_distill = (self.mse_loss(P_delta_text, Delta_text) +
                                    self.mse_loss(P_delta_image, Delta_image) +
                                    self.mse_loss(P_delta_cross, Delta_cross)) / 3.0

                loss_manip = torch.tensor(0.0, device=labels.device)
                if manip_logits is not None and self.lambda_manipulation_predict > 0:
                    manip_labels = batch_data.get('manipulation_labels')
                    if manip_labels is not None:
                        loss_manip = self.ce_loss(manip_logits, manip_labels.long())

                total_loss = loss_detection + self.lambda_reasoning_align * loss_distill + self.lambda_manipulation_predict * loss_manip

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if self.scheduler: self.scheduler.step()
                avg_loss.add(total_loss.item())
                train_iter.set_postfix(loss=avg_loss.item())

            print(f'Epoch {epoch + 1} Loss: {avg_loss.item():.4f}')
            val_results = self.test(self.val_loader)
            mark = recorder.add(val_results)
            if mark == 'save':
                torch.save(self.model.state_dict(), os.path.join(self.save_param_dir, self.model_save_filename))
            elif mark == 'esc':
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, self.model_save_filename)))
        test_results = self.test(self.test_loader)
        print(f"Final Test Results: {test_results}")
        return test_results, os.path.join(self.save_param_dir, self.model_save_filename)

    def test(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_categories = [], [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch_data = clipdata2gpu(batch, self.use_cuda)
                final_prob, _, _, _, _, _, _, _, _, _, _ = self.model(**batch_data)

                all_preds.extend(final_prob.cpu().numpy())
                all_labels.extend(batch_data['label'].cpu().numpy())
                if 'category' in batch_data:
                    all_categories.extend(batch_data['category'].cpu().numpy())

        return metricsTrueFalse(all_labels, all_preds, all_categories, self.category_dict)

# Author: 
# Corresponding Mail: 
