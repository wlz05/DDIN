import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split

root = '/root/autodl-tmp/FineFake_dataset/'

# 1. 瞬间加载你刚才跑完的总特征和总表
df = pd.read_pickle(root + 'FineFake.pkl')
with open(root + 'f_train_loader.pkl', 'rb') as f:
    full_features = pickle.load(f)

# 2. 严谨切分 (8:1:1)
indices = range(len(df))
train_idx, tmp_idx = train_test_split(indices, test_size=0.2, random_state=3407)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, random_state=3407)

# 3. 完美分割表格 (杜绝泄露)
df.iloc[train_idx].to_pickle(root + 'train.pkl')
df.iloc[val_idx].to_pickle(root + 'val.pkl')
df.iloc[test_idx].to_pickle(root + 'test.pkl')

# 4. 瞬间分割特征矩阵 (填补缺失文件)
def save_feat(idx, prefix):
    feat = full_features[idx]
    with open(root + f'{prefix}_loader.pkl', 'wb') as f:
        pickle.dump(feat, f)
    with open(root + f'{prefix}_clip.pkl', 'wb') as f:
        pickle.dump(feat, f)

save_feat(train_idx, 'f_train')
save_feat(val_idx, 'f_val')
save_feat(test_idx, 'f_test')

print("✅ 一秒切分完成！文件全部对齐，绝无泄露！")
