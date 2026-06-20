# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

# Usage: python feature.py --dataset weibo --batchsize 64

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import logging

# --- 1. Imports ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from model.weibo import MultiDomainPLEFENDModel
    logger.info("Imported MultiDomainPLEFENDModel from model.weibo.")
except ImportError as e:
    logger.warning(f"MultiDomainPLEFENDModel not available: {e}.")
    logger.warning("feature.py requires model/weibo.py. Run main.py first for training.")
    MultiDomainPLEFENDModel = None
try:
    from run import Run
    logger.info("Imported Run class from run.py.")
except ImportError as e:
    logger.error(f"Failed to import Run class: {e}")
    exit()

# --- 2. Args ---
parser = argparse.ArgumentParser(description="Generate t-SNE visualization for a dataset.")
parser.add_argument('--dataset', default='weibo', choices=['weibo', 'weibo21'], help="Dataset to visualize.")
parser.add_argument('--batchsize', type=int, default=64, help="Batch size for data loading.")
parser.add_argument('--gpu', default='0', help="GPU device ID.")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- 3. Config & DataLoader ---
logger.info(f"Building config for dataset '{args.dataset}'...")
if args.dataset == 'weibo':
    config = {
        'dataset': 'weibo', 'model_name': 'domain_weibo', 'weibo_data_dir': './data/',
        'bert_model_path_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
        'bert_vocab_file_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'batchsize': args.batchsize, 'max_len': 197, 'num_workers': 4, 'emb_dim': 768,
        'lr': 0.000175, 'early_stop': 100, 'epoch': 50, 'save_param_dir': './param_model',
        # MLP dims set to [384] to match saved weights
        'model_params': {'mlp': {'dims': [384], 'dropout': 0.2}},
        'vocab_file': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'bert': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
    }
elif args.dataset == 'weibo21':
    # weibo21 already uses [384]
    config = {
        'dataset': 'weibo21', 'model_name': 'domain_weibo', 'weibo21_data_dir': './w21/',
        'bert_model_path_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
        'bert_vocab_file_weibo': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'batchsize': args.batchsize, 'max_len': 197, 'num_workers': 4, 'emb_dim': 768,
        'lr': 0.0005, 'early_stop': 100, 'epoch': 50, 'save_param_dir': './param_model',
        'model_params': {'mlp': {'dims': [384], 'dropout': 0.2}},
        'vocab_file': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt',
        'bert': './pretrained_model/chinese_roberta_wwm_base_ext_pytorch',
    }
else:
    raise ValueError("This script only supports 'weibo' and 'weibo21' datasets.")

# --- 4 & 5. Model Loading & Feature Extraction ---
logger.info("Initializing Run class to get data loader...")
try:
    runner = Run(config)
    _, _, test_loader = runner.get_dataloader()
    logger.info("Got test_loader successfully.")
except Exception as e:
    logger.error(f"Error getting dataloader: {e}", exc_info=True)
    exit()

logger.info("Initializing model...")
final_mlp_dims = config['model_params']['mlp']['dims'] + [1]
logger.info(f"MLP dims for {args.dataset}: {final_mlp_dims}")

model = MultiDomainPLEFENDModel(
    emb_dim=config['emb_dim'],
    mlp_dims=final_mlp_dims,
    bert=config['bert'],
    out_channels=320,  # matches domain_weibo.py
    dropout=config['model_params']['mlp']['dropout']
).to(DEVICE)

try:
    MODEL_WEIGHTS_PATH = os.path.join(config['save_param_dir'], f"{config['dataset']}_{config['model_name']}",
                                      'parameter_calibration_distill.pkl')
    logger.info(f"Loading weights from: {MODEL_WEIGHTS_PATH}")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Weight file not found: '{MODEL_WEIGHTS_PATH}'.")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE), strict=False)
    logger.info("Weights loaded successfully (non-strict mode)!")
except Exception as e:
    logger.error(f"Failed to load weights: {e}")
    exit()

model.eval()
logger.info("Extracting features...")
all_features, all_labels = [], []
with torch.no_grad():
    try:
        from utils.utils_weibo import clipdata2gpu as c2g
    except ImportError:
        from utils.utils import clipdata2gpu as c2g
        logger.warning("utils_weibo not found, using utils.utils.clipdata2gpu")

    for batch in tqdm(test_loader, desc="Extracting features"):
        batch_data = c2g(batch) if DEVICE.type == 'cuda' else batch

        # model.forward() returns an 11-element tuple
        model_outputs = model(**batch_data)

        # Extract F_text (idx 4), F_image (idx 5), F_cross (idx 6)
        f_text = model_outputs[4]
        f_image = model_outputs[5]
        f_cross = model_outputs[6]

        features = torch.cat((f_text, f_image, f_cross), dim=1)

        labels = batch_data['label']
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
logger.info(f"Feature extraction complete: {all_features.shape[0]} samples.")

# --- Save features ---
output_dir = './extracted_features'
os.makedirs(output_dir, exist_ok=True)
feature_filename = os.path.join(output_dir, f'features_{args.dataset}.npz')
logger.info(f"Saving features and labels to: {feature_filename}")
np.savez_compressed(feature_filename, features=all_features, labels=all_labels)
logger.info("Features and labels saved!")

# --- 6. t-SNE ---
logger.info("Running t-SNE dimensionality reduction...")
tsne = TSNE(
    n_components=2, verbose=1, perplexity=15, early_exaggeration=15,
    n_iter=700, learning_rate='auto', init='pca', random_state=19
)
tsne_results = tsne.fit_transform(all_features)
logger.info("t-SNE complete!")

# --- 7. Plot ---
logger.info("Plotting visualization...")
fig, ax = plt.subplots(figsize=(8, 8))
custom_palette = ["#E67E22", "#3498DB"]  # orange vs blue

sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=all_labels, palette=custom_palette,
    s=80, alpha=0.85, linewidth=0, legend=False, ax=ax
)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_ylabel(config['dataset'].capitalize(), rotation='vertical', fontsize=28, labelpad=20)

output_dir = './visualizations'
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f'tsne_{config["dataset"]}_more_overlap.png')
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
logger.info(f"Visualization saved to: {output_filename}")
plt.show()
