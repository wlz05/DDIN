# DDIN: Domain-aware Disentangled Interaction Network for Multimodal Fake News Detection

import os
from utils.clipld import bert_data as weibo_data
from utils.w21ld import bert_data as weibo21_data
from utils.fld import bert_data as finefake_data
from model.net import Trainer as DDINTrainer
from model.gossip import Trainer as GossipTrainer

class Run():
    def __init__(self, config):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']

        if config['dataset'] == "weibo":
            self.root_path = './data/'
            self.train_path = self.root_path + 'train_origin.csv'
            self.val_path = self.root_path + 'val_origin.csv'
            self.test_path = self.root_path + 'test_origin.csv'
            self.category_dict = {
                "Economy": 0, "Health": 1, "Military": 2, "Science": 3,
                "Politics": 4, "International": 5, "Education": 6, "Entertainment": 7, "Society": 8
            }
        elif config['dataset'] == "weibo21":
            self.root_path = './Weibo21/'
            self.train_path = self.root_path + 'train_datasets.xlsx'
            self.val_path = self.root_path + 'val_datasets.xlsx'
            self.test_path = self.root_path + 'test_datasets.xlsx'
            self.category_dict = {
                "Technology": 0, "Military": 1, "Education": 2, "Disaster": 3,
                "Politics": 4, "Healthcare": 5, "Finance": 6, "Entertainment": 7, "Society": 8
            }
        elif config['dataset'] == "finefake":
            self.root_path = './FineFake/'
            self.train_path = self.root_path + config.get('finefake_train', 'gossip_train.csv')
            self.val_path = self.root_path + config.get('finefake_val', 'gossip_test.csv')
            self.test_path = self.root_path + config.get('finefake_test', 'gossip_test.csv')
            self.category_dict = {
                "Politics": 0, "Entertainment": 1, "Business": 2,
                "Health": 3, "Society": 4, "Conflict": 5
            }

    def get_dataloader(self, dataset):
        loader = None
        if self.emb_type == 'bert':
            if dataset == "weibo":
                loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                    category_dict=self.category_dict, num_workers=self.num_workers)
            elif dataset == "weibo21":
                loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                      category_dict=self.category_dict, num_workers=self.num_workers)
            elif dataset == "finefake":
                loader = finefake_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                       category_dict=self.category_dict, num_workers=self.num_workers)

        if dataset == "weibo":
            train_loader = loader.load_data(self.train_path, 'data/train_loader.pkl', 'data/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, 'data/val_loader.pkl', 'data/val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, 'data/test_loader.pkl', 'data/test_clip_loader.pkl', False)
        elif dataset == "weibo21":
            train_loader = loader.load_data(self.train_path, 'Weibo21/train_loader.pkl', 'Weibo21/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, 'Weibo21/val_loader.pkl', 'Weibo21/val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, 'Weibo21/test_loader.pkl', 'Weibo21/test_clip_loader.pkl', False)
        elif dataset == "finefake":
            ff_dir = self.root_path
            train_loader = loader.load_data(self.train_path, ff_dir + 'f_train_loader.pkl',
                                            ff_dir + 'f_train_clip.pkl', True)
            val_loader = loader.load_data(self.val_path, ff_dir + 'f_val_loader.pkl',
                                          ff_dir + 'f_val_clip.pkl', False)
            test_loader = loader.load_data(self.test_path, ff_dir + 'f_test_loader.pkl',
                                           ff_dir + 'f_test_clip.pkl', False)

        return train_loader, val_loader, test_loader

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)

        if self.model_name == 'DDIN':
            trainer = DDINTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                                    use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader, dropout=self.dropout,
                                    weight_decay=self.weight_decay, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name))
            trainer.train()
        elif self.model_name == 'Gossip':
            trainer = GossipTrainer(emb_dim=self.emb_dim, mlp_dims=self.mlp_dims,
                                    bert_path_or_name=self.bert,
                                    clip_path_or_name='ViT-B-16',
                                    use_cuda=self.use_cuda, lr=self.lr, dropout=self.dropout,
                                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                    category_dict=self.category_dict, weight_decay=self.weight_decay,
                                    save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                                    early_stop=self.early_stop, epoches=self.epoch)
            trainer.train()
