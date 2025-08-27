import torch
import torch.nn as nn
import numpy as np
import functools
from random import sample

from ..subNets.transformers_encoder.transformer import TransformerEncoder
from ..subNets.score_sde.sde_utils import create_score_model
from ..subNets.score_sde import sde_lib, losses
from ..subNets.score_sde import ddpm, ncsnv2, ncsnpp
from ..subNets.score_sde.sampling import get_sampling_fn
from ..subNets.score_sde.rcan import Group

from ..subNets import AlignSubNet

from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer, T5Model, T5Tokenizer, DebertaV2Tokenizer, DebertaV2Model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
    'deberta': (DebertaV2Model, DebertaV2Tokenizer),
    'xlnet': (XLNetModel, XLNetTokenizer),
    't5': (T5Model, T5Tokenizer),
    'sbert': (AutoModel, AutoTokenizer)
}

__all__ = ['ABDiff']

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class AffectPred(nn.Module):
    def __init__(self, in_size, n_class, dropout):
        '''
        Args:
            in_size: input dimension
            dropout: dropout probability
        '''
        super(AffectPred, self).__init__() #SubNet
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size)
        self.linear_2 = nn.Linear(in_size, in_size)
        self.linear_3 = nn.Linear(in_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        y_1 = torch.relu(self.linear_1(x))
        y_h = self.linear_2(self.drop(y_1))
        y_h += x # resdual block
        y = self.linear_3(y_h)
        return y

# thanks to https://github.com/udeepam/vib
class ConditionIB(nn.Module):
    def __init__(self, input_shape, output_shape, bottleneck_dim):
        """
        Condition Information Bottleneck Model.
        
        Arguments:
        ----------
        input_shape : `int`
            Flattened size of raw input.
        output_shape : `int`
            Number of classes.        
        bottleneck_dim : `int`
            The dimension of the latent bottleneck condition. (Default=256)
        """        
        super(ConditionIB, self).__init__()
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.bottleneck_dim  = bottleneck_dim
        mlp_dim  = 256

        # build encoder
        self.encoder = nn.Sequential(nn.Linear(input_shape, mlp_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(mlp_dim, mlp_dim),
                                     nn.ReLU(inplace=True))
        self.fc_mu  = nn.Linear(mlp_dim, bottleneck_dim)
        self.fc_std = nn.Linear(mlp_dim, bottleneck_dim)
        # self.encoder = nn.Sequential(nn.Conv1d(input_shape, 1024, kernel_size=1, padding=0),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv1d(1024, 1024, kernel_size=1, padding=0),
        #                              nn.ReLU(inplace=True))
        # self.fc_mu  = nn.Conv1d(1024, bottleneck_dim, kernel_size=1, padding=0)
        # self.fc_std = nn.Conv1d(1024, bottleneck_dim, kernel_size=1, padding=0)
        
        # build decoder
        self.decoder = nn.Linear(bottleneck_dim, output_shape)
        # self.decoder = nn.Conv1d(bottleneck_dim, output_shape, kernel_size=1, padding=0)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), torch.nn.functional.softplus(self.fc_std(x)-5, beta=1)
    
    def decode(self, c):
        """
        c : [batch_size,bottleneck_dim]
        """ 
        return self.decoder(c)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,bottleneck_dim]
        std : [batch_size,bottleneck_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps
    
    def KL_div(self, mu, std):
        """    
        Loss IB distribution divergence between input and bottleneck: beta*KL divergence

        mu : [batch_size,bottleneck_dim]  
        std: [batch_size,bottleneck_dim] 
        """
        beta = 1e-3   
        KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (beta*KL) #/ mu.size(0)

    def forward(self, x):
        """
        Forward pass 
        
        Parameters: x : [batch_size, dimension, sequence_length]
        """
        B, D, L = x.shape
        x_flat = x.contiguous().view(B, -1)
        mu, std = self.encode(x_flat)
        c_flat = self.reparameterise(mu, std)
        c = c_flat.reshape(B, self.bottleneck_dim, -1)
        # return c, self.decode(c), mu, std
        return c, self.decode(c_flat), self.KL_div(mu, std) # condition, prediction, KL_div

class ABDiff(nn.Module):
    def __init__(self, args):
        super(ABDiff, self).__init__()
        self.args = args
        self.device = args.device
        self.aligned = args.need_data_aligned

        self.text_out, self.audio_out, self.vision_out = args.feature_dims
        self.common_out = args.common_out
        self.layers = args.nlevels
        self.num_heads = args.num_heads
        self.attn_dropout_l = args.attn_dropout_l
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_mem = args.attn_dropout_mem
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        self.output_dropout = args.output_dropout

        tokenizer_class = TRANSFORMERS_MAP[args.transformers][1]
        model_class = TRANSFORMERS_MAP[args.transformers][0]
        self.text_tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_name_or_path='/presearch_lin/AffectiveComputing/pretrains/' + args.pretrained,
            do_lower_case=True)
        self.text_model = model_class.from_pretrained(
            pretrained_model_name_or_path='/presearch_lin/AffectiveComputing/pretrains/' + args.pretrained)
        print(f"Based on Language Model with {sum(p.numel() for p in self.text_model.parameters())} parameters! ")
        if self.args.generation_stage:
            self.text_model.eval()
            self.text_model.requires_grad_(False)
        self.max_text_length = self.args.max_text_length

        self.alignNet = AlignSubNet(args, mode='avg_pool') # mode in ['avg_pool', 'ctc', 'conv1d']

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.text_out, self.common_out, kernel_size=args.conv1d_kernel_size_l, padding=args.padding, bias=False)
        self.proj_a = nn.Conv1d(self.audio_out, self.common_out, kernel_size=args.conv1d_kernel_size_a, padding=args.padding, bias=False)
        self.proj_v = nn.Conv1d(self.vision_out, self.common_out, kernel_size=args.conv1d_kernel_size_v, padding=args.padding, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')

        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')

        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')

        # 3. Self Attentions
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        output_dim = args.num_classes if args.train_mode in ("detection", "recognition") else 1
        self.proj_lav = nn.Linear(2* 3 * self.common_out, self.common_out)
        self.fusion_pred_head_final = AffectPred(in_size=self.common_out, n_class=output_dim, dropout=self.output_dropout)

        self.cond_out = self.common_out
        self.adopt_IB = args.adopt_IB
        # Projection layers
        if args.generation_stage:
            if self.adopt_IB == True:
                # Information Bottleneck Restricted Condition Projecting
                # self.cond_proj_lav = ConditionIB(np.prod(args.batch_size,args.data_seq_len,3*self.common_out), output_dim, self.cond_out)
                self.cond_proj_av = ConditionIB(args.data_seq_len*2*self.common_out, output_dim, self.cond_out)
                self.cond_proj_al = ConditionIB(args.data_seq_len*2*self.common_out, output_dim, self.cond_out)
                self.cond_proj_lv = ConditionIB(args.data_seq_len*2*self.common_out, output_dim, self.cond_out)
                self.cond_proj_l = ConditionIB(args.data_seq_len*self.common_out, output_dim, self.cond_out)
                self.cond_proj_a = ConditionIB(args.data_seq_len*self.common_out, output_dim, self.cond_out)
                self.cond_proj_v = ConditionIB(args.data_seq_len*self.common_out, output_dim, self.cond_out)
            else:
                # Condition Projecting
                # self.cond_proj_lav = nn.Conv1d(3 * self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_av = nn.Conv1d(2 * self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_al = nn.Conv1d(2 * self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_lv = nn.Conv1d(2 * self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_l = nn.Conv1d(self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_a = nn.Conv1d(self.common_out, self.cond_out, kernel_size=1, padding=0)
                self.cond_proj_v = nn.Conv1d(self.common_out, self.cond_out, kernel_size=1, padding=0)

                # self.fusion_pred_head_lav = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_av = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_al = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_lv = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_l = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_a = nn.Linear(self.cond_out, output_dim)
                self.fusion_pred_head_v = nn.Linear(self.cond_out, output_dim)

            self.l_args = self.a_args = self.v_args = self.args
            self.l_args.score_sigma_max, self.l_args.score_sigma_min = self.args.score_sigma_l_max, self.args.score_sigma_l_max
            self.a_args.score_sigma_max, self.a_args.score_sigma_min = self.args.score_sigma_a_max, self.args.score_sigma_a_min
            self.v_args.score_sigma_max, self.v_args.score_sigma_min = self.args.score_sigma_v_max, self.args.score_sigma_v_min
            self.score_l = create_score_model(self.args.score_model_name, self.l_args)
            self.score_a = create_score_model(self.args.score_model_name, self.a_args)
            self.score_v = create_score_model(self.args.score_model_name, self.v_args)

            # Setup SDEs
            if self.args.sde.lower() == 'vpsde':
                sde = sde_lib.VPSDE(beta_min=self.args.score_beta_min, beta_max=self.args.score_beta_max, N=self.args.score_num_scales)
                sampling_eps = 1e-3
            elif self.args.sde.lower() == 'subvpsde':
                sde = sde_lib.subVPSDE(beta_min=self.args.score_beta_min, beta_max=self.args.score_beta_max, N=self.args.score_num_scales)
                sampling_eps = 1e-3
            elif self.args.sde.lower() == 'vesde':
                sde_l = sde_lib.VESDE(sigma_min=self.args.score_sigma_l_min, sigma_max=self.args.score_sigma_l_max, N=self.args.score_num_scales)
                sde_a = sde_lib.VESDE(sigma_min=self.args.score_sigma_a_min, sigma_max=self.args.score_sigma_a_max, N=self.args.score_num_scales)
                sde_v = sde_lib.VESDE(sigma_min=self.args.score_sigma_v_min, sigma_max=self.args.score_sigma_v_max, N=self.args.score_num_scales)
                sampling_eps = 1e-5
            else:
                raise NotImplementedError(f"SDE {self.args.sde} unknown.")

            self.score_loss_func_l = losses.get_loss_fn(sde_l, train=True,
                                     reduce_mean=self.args.sde_reduce_mean, continuous=self.args.sde_continuous,
                                     likelihood_weighting=self.args.sde_likelihood_weighting)
            self.score_loss_func_a = losses.get_loss_fn(sde_a, train=True,
                                     reduce_mean=self.args.sde_reduce_mean, continuous=self.args.sde_continuous,
                                     likelihood_weighting=self.args.sde_likelihood_weighting)
            self.score_loss_func_v = losses.get_loss_fn(sde_v, train=True,
                                     reduce_mean=self.args.sde_reduce_mean, continuous=self.args.sde_continuous,
                                     likelihood_weighting=self.args.sde_likelihood_weighting)
            
            sampling_shape = (self.args.batch_size, self.args.data_num_channel, self.args.data_seq_len)
            # (self.args.eval_batch_size, self.args.data_num_channel,
            #          self.args.data_seq_len) if 'eval_batch_size' in args else
            
            inverse_scaler = lambda x: x # Inverse data normalizer: lambda x: (x + 1.) / 2. for self.centered=True 
            self.Sampler_l = get_sampling_fn(self.l_args, sde_l, sampling_shape, inverse_scaler, sampling_eps)
            self.Sampler_a = get_sampling_fn(self.a_args, sde_a, sampling_shape, inverse_scaler, sampling_eps)
            self.Sampler_v = get_sampling_fn(self.v_args, sde_v, sampling_shape, inverse_scaler, sampling_eps)

    def preprocess_pad(self, x, dim =1):
        if x.size(dim=1) % 2 == 1: # if dim L is odd
            x = nn.functional.pad(x, (0,0,0,1), mode='constant', value=0)
        return x

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.common_out, self.attn_dropout_l
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.common_out, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.common_out, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.common_out, self.attn_dropout_mem
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.common_out, self.attn_dropout_mem
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.common_out, self.attn_dropout_mem
        else:
            raise ValueError("Unknown network type")

        # TODO: Replace with nn.TransformerEncoder
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def add_gaussian_noise(self, tensor, mean=0., std=1.):
        noise = torch.randn(tensor.size()) * std + mean
        noisy_tensor = tensor + noise.to(self.device)
        return noisy_tensor 

    def normalization(self, x):
        eps = 1e-5 
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + eps), norm
    
    def unnormalization(self, x_norm, norm):
        return x_norm * norm
    
    def get_mean_std(self, x):
        B, D, L = x.shape
        x_vec = x.transpose(1, 2).contiguous().view(-1, D) # N*L, d_L
        # x_vec = x.mean(dim=-1) # N, d_L
        x_mean = torch.mean(x_vec, dim=0)
        x_std = torch.std(x_vec-x_mean, unbiased=False)
        return x_mean, x_std

    # thanks to https://github.com/justinlovelace/latent-diffusion-for-language
    def normalize_latent(self, x_start, latent_mean, latent_std):
        eps = 1e-5 
        # https://doi.org/10.1007/978-0-387-32833-1_54
        return (x_start-latent_mean.unsqueeze(dim=1))/(latent_std).clamp(min=eps)
    
    def unnormalize_latent(self, x_start, latent_mean, latent_std):
        eps = 1e-5 
        return (x_start*(latent_std.clamp(min=eps))+latent_mean.unsqueeze(dim=1))
    
    def cosine_similarity(self, a, b, eps=1e-8):
        """计算两个向量的余弦相似度 (逐样本)"""
        dot = torch.sum(a * b, dim=-1)  # 注意这里用 dim
        norm_a = torch.norm(a, dim=-1)
        norm_b = torch.norm(b, dim=-1)
        return torch.mean(dot / (norm_a * norm_b + eps))
    
    def make_gmm_1d(self, n, means=(-3.0, 3.0), stds=(0.7, 0.7), weights=(0.5, 0.5), device="cpu"):
        """Sample n points from a 1D Gaussian mixture."""
        assert abs(sum(weights) - 1.0) < 1e-6
        comps = np.random.choice(len(means), size=n, p=weights)
        x = np.random.randn(n) * np.array(stds)[comps] + np.array(means)[comps]
        return torch.from_numpy(x).float().to(device).unsqueeze(1)  # shape: [n, 1]

    def make_gmm_tensor_like(self, proj_x_l, means=(-3.0, 3.0), stds=(0.7, 0.7), weights=(0.5, 0.5)):
        """
        给定 shape [b, l, d] 的 tensor proj_x_l，
        生成相同 shape 的 GMM 样本张量。
        """
        device = proj_x_l.device
        b, l, d = proj_x_l.shape
        n = b * l * d
        gmm_samples = self.make_gmm_1d(n, means, stds, weights, device=device)  # [n, 1]
        gmm_samples = gmm_samples.view(b, l, d)  # reshape 回去

        return gmm_samples
    
    def forward(self, text, audio, vision, y=None, criterion_func=None, mask_matrix=None): # , gen_epoch=False, mean_std=None
        if self.args.generation_stage:
            with torch.no_grad():
                _text_z = self.text_tokenizer(text,
                                                add_special_tokens=True,
                                                max_length=self.max_text_length, padding='max_length', truncation=True,
                                                return_tensors='pt').to(self.device)
                text = self.text_model(**_text_z).last_hidden_state

            with torch.no_grad():
                text, audio, vision = self.alignNet(text, audio, vision)

                x_l = text.transpose(1, 2)
                x_a = audio.transpose(1, 2)
                x_v = vision.transpose(1, 2)

                # Project the textual/visual/audio features
                proj_x_l = x_l if self.text_out == self.common_out else self.proj_l(x_l) # Dimension (N, d_l, L)
                proj_x_a = x_a if self.audio_out == self.common_out else self.proj_a(x_a)
                proj_x_v = x_v if self.vision_out == self.common_out else self.proj_v(x_v)

            # original IMDer
            gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a

            # # normalization of latent (better for generation)
            # latent_mean_l, latent_std_l = self.get_mean_std(proj_x_l)
            # latent_mean_v, latent_std_v = self.get_mean_std(proj_x_v)
            # latent_mean_a, latent_std_a = self.get_mean_std(proj_x_a)
            # proj_x_l = self.normalize_latent(proj_x_l, latent_mean_l, latent_std_l)
            # proj_x_a = self.normalize_latent(proj_x_a, latent_mean_a, latent_std_a)
            # proj_x_v = self.normalize_latent(proj_x_v, latent_mean_v, latent_std_v)

            if self.adopt_IB == True:
                # Tri-modal/Bi-modal/Uni-modal Condition and Prediction
                # cond_lav, output_lav, loss_ib_lav = self.cond_proj_lav(torch.cat([proj_x_l, proj_x_a, proj_x_v], dim=1))
                cond_av, output_av, loss_ib_av = self.cond_proj_av(torch.cat([proj_x_a, proj_x_v], dim=1))
                cond_lv, output_lv, loss_ib_lv = self.cond_proj_lv(torch.cat([proj_x_l, proj_x_v], dim=1))
                cond_al, output_al, loss_ib_al = self.cond_proj_al(torch.cat([proj_x_a, proj_x_l], dim=1))
                cond_l, output_l, loss_ib_l = self.cond_proj_l(proj_x_l)
                cond_a, output_a, loss_ib_a = self.cond_proj_a(proj_x_a)
                cond_v, output_v, loss_ib_v = self.cond_proj_v(proj_x_v)
            else:
                # Tri-modal/Bi-modal/Uni-modal Condition
                # cond_lav = self.cond_proj_lav(torch.cat([proj_x_l, proj_x_a, proj_x_v], dim=1))
                cond_av = self.cond_proj_av(torch.cat([proj_x_a, proj_x_v], dim=1))
                cond_lv = self.cond_proj_lv(torch.cat([proj_x_l, proj_x_v], dim=1))
                cond_al = self.cond_proj_al(torch.cat([proj_x_a, proj_x_l], dim=1))
                cond_l = self.cond_proj_l(proj_x_l)
                cond_a = self.cond_proj_a(proj_x_a)
                cond_v = self.cond_proj_v(proj_x_v)
                # Condition Prediction
                # output_lav = self.fusion_pred_head_lav(cond_lav.mean(dim=-1))
                output_av = self.fusion_pred_head_av(cond_av.mean(dim=-1))
                output_lv = self.fusion_pred_head_lv(cond_lv.mean(dim=-1))
                output_al = self.fusion_pred_head_al(cond_al.mean(dim=-1))
                output_l = self.fusion_pred_head_l(cond_l.mean(dim=-1))
                output_a = self.fusion_pred_head_a(cond_a.mean(dim=-1))
                output_v = self.fusion_pred_head_v(cond_v.mean(dim=-1))
                
            ''' Condition-based Score Model '''
            loss_score = torch.tensor(0., dtype=torch.float32).to(self.device)
            loss_ib = torch.tensor(0., dtype=torch.float32).to(self.device)
            loss_cond = torch.tensor(0., dtype=torch.float32).to(self.device)
            loss_dec = torch.tensor(0., dtype=torch.float32).to(self.device)
            if self.training:

                for condition in [cond_a, cond_v, cond_av]:
                    loss_score += self.score_loss_func_l(self.score_l, proj_x_l, condition=condition)
                for condition in [cond_l, cond_v, cond_lv]:
                    loss_score += self.score_loss_func_a(self.score_a, proj_x_a, condition=condition)
                for condition in [cond_a, cond_l, cond_al]:
                    loss_score += self.score_loss_func_v(self.score_v, proj_x_v, condition=condition)
                loss_score = 1/9 * loss_score

                # Calculate IB loss for condition
                if self.adopt_IB == True:
                    loss_ib += 1/6 * (loss_ib_av + loss_ib_lv + loss_ib_al + loss_ib_l + loss_ib_a + loss_ib_v)

                loss_cond += 1/6 * (criterion_func(output_av, y) + criterion_func(output_lv, y) + \
                            criterion_func(output_al, y) + criterion_func(output_l, y) + criterion_func(output_a, y) + \
                            criterion_func(output_v, y))

            else:
                assert mask_matrix != None, "Error: no mask matrix setting!!!"
                mask_l = mask_matrix[:, 0]  # [B]
                mask_a = mask_matrix[:, 1]  # [B]
                mask_v = mask_matrix[:, 2]  # [B]
                
                proj_x_l_gen_a, n = self.Sampler_a(self.score_a, condition=cond_l) # (N, d_l, L)
                proj_x_l_gen_v, n = self.Sampler_v(self.score_v, condition=cond_l)
                proj_x_a_gen_l, n = self.Sampler_l(self.score_l, condition=cond_a)
                proj_x_a_gen_v, n = self.Sampler_v(self.score_v, condition=cond_a)
                proj_x_v_gen_l, n = self.Sampler_l(self.score_l, condition=cond_v)
                proj_x_v_gen_a, n = self.Sampler_a(self.score_a, condition=cond_v)
                proj_x_lv_gen_a, n = self.Sampler_a(self.score_a, condition=cond_lv)
                proj_x_al_gen_v, n = self.Sampler_v(self.score_v, condition=cond_al)
                proj_x_av_gen_l, n = self.Sampler_l(self.score_l, condition=cond_av)

                mask_text_av = ((mask_l == 0) & (mask_a == 1) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # audio 和 vision 均可用
                mask_text_a  = ((mask_l == 0) & (mask_a == 1) & (mask_v == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 audio 可用
                mask_text_v  = ((mask_l == 0) & (mask_a == 0) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 vision 可用
                proj_x_l = torch.where(mask_text_av, proj_x_av_gen_l, proj_x_l)
                proj_x_l = torch.where(mask_text_a,  proj_x_a_gen_l,  proj_x_l)
                proj_x_l = torch.where(mask_text_v,  proj_x_v_gen_l,  proj_x_l)

                mask_audio_lv = ((mask_a == 0) & (mask_l == 1) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # text 和 vision 均可用
                mask_audio_l  = ((mask_a == 0) & (mask_l == 1) & (mask_v == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 text 可用
                mask_audio_v  = ((mask_a == 0) & (mask_l == 0) & (mask_v == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 vision 可用
                proj_x_a = torch.where(mask_audio_lv, proj_x_lv_gen_a, proj_x_a)
                proj_x_a = torch.where(mask_audio_l,  proj_x_l_gen_a,  proj_x_a)
                proj_x_a = torch.where(mask_audio_v,  proj_x_v_gen_a,  proj_x_a)

                mask_vision_la = ((mask_v == 0) & (mask_l == 1) & (mask_a == 1)).unsqueeze(-1).unsqueeze(-1)  # text 和 audio 均可用
                mask_vision_l  = ((mask_v == 0) & (mask_l == 1) & (mask_a == 0)).unsqueeze(-1).unsqueeze(-1)  # 仅 text 可用
                mask_vision_a  = ((mask_v == 0) & (mask_l == 0) & (mask_a == 1)).unsqueeze(-1).unsqueeze(-1)  # 仅 audio 可用
                proj_x_v = torch.where(mask_vision_la, proj_x_al_gen_v, proj_x_v)
                proj_x_v = torch.where(mask_vision_l,  proj_x_l_gen_v,  proj_x_v)
                proj_x_v = torch.where(mask_vision_a,  proj_x_a_gen_v,  proj_x_v)

            # # unnormalization of latent
            # proj_x_l = self.unnormalize_latent(proj_x_l, latent_mean_l, latent_std_l)
            # proj_x_a = self.unnormalize_latent(proj_x_a, latent_mean_a, latent_std_a)
            # proj_x_v = self.unnormalize_latent(proj_x_v, latent_mean_v, latent_std_v)

            with torch.no_grad():
                ''' Multimodal Fusion based on MulT '''
                proj_x_a = proj_x_a.permute(2, 0, 1)  # Dimension (L, N, d_l)
                proj_x_v = proj_x_v.permute(2, 0, 1)
                proj_x_l = proj_x_l.permute(2, 0, 1)
                # (V,A) --> L
                h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) # Dimension (L, N, d_l)
                h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
                h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=-1)
                h_ls = self.trans_l_mem(h_ls)
                if type(h_ls) == tuple:
                    h_ls = h_ls[0]
                last_h_l = h_ls #.mean(dim=0) #= h_ls[-1]  # Take the last output for prediction

                # (L,V) --> A
                h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
                h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
                h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=-1)
                h_as = self.trans_a_mem(h_as)
                if type(h_as) == tuple:
                    h_as = h_as[0]
                last_h_a = h_as #.mean(dim=0) #= h_as[-1]

                # (L,A) --> V
                h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
                h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
                h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=-1)
                h_vs = self.trans_v_mem(h_vs)
                if type(h_vs) == tuple:
                    h_vs = h_vs[0]
                last_h_v = h_vs #.mean(dim=0) #= h_vs[-1]

                # Final Representation
                final_lav = torch.cat([last_h_l, last_h_a, last_h_v], dim=-1) # Dimension (L, N, d_l)
                final_f = self.proj_lav(final_lav).permute(1, 2, 0) # Dimension (N, d_l, L)

                # Final Prediction
                output_f = self.fusion_pred_head_final(final_f.mean(dim=-1))

        else:
            _text_z = self.text_tokenizer(text,
                                            add_special_tokens=True,
                                            max_length=self.max_text_length, padding='max_length', truncation=True,
                                            return_tensors='pt').to(self.device)
            text = self.text_model(**_text_z).last_hidden_state
        
            text, audio, vision = self.alignNet(text, audio, vision)

            x_l = text.transpose(1, 2)
            x_a = audio.transpose(1, 2)
            x_v = vision.transpose(1, 2)

            # Project the textual/visual/audio features
            proj_x_l = x_l if self.text_out == self.common_out else self.proj_l(x_l) # Dimension (N, d_l, L)
            proj_x_a = x_a if self.audio_out == self.common_out else self.proj_a(x_a)
            proj_x_v = x_v if self.vision_out == self.common_out else self.proj_v(x_v)

            # original IMDer
            gt_l, gt_v, gt_a = proj_x_l, proj_x_v, proj_x_a
            
            ''' Multimodal Fusion based on MulT '''
            proj_x_a = proj_x_a.permute(2, 0, 1)  # Dimension (L, N, d_l)
            proj_x_v = proj_x_v.permute(2, 0, 1)
            proj_x_l = proj_x_l.permute(2, 0, 1)
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a) # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=-1)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls #.mean(dim=0) #= h_ls[-1]  # Take the last output for prediction

            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=-1)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as #.mean(dim=0) #= h_as[-1]

            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=-1)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = h_vs #.mean(dim=0) #= h_vs[-1]

            # Final Representation
            final_lav = torch.cat([last_h_l, last_h_a, last_h_v], dim=-1) # Dimension (L, N, d_l)
            final_f = self.proj_lav(final_lav).permute(1, 2, 0) # Dimension (N, d_l, L)

            # Final Prediction
            output_f = self.fusion_pred_head_final(final_f.mean(dim=-1))
            loss_score = torch.tensor(0., dtype=torch.float32).to(self.device)
            loss_ib = torch.tensor(0., dtype=torch.float32).to(self.device)
            loss_cond = torch.tensor(0., dtype=torch.float32).to(self.device)

        res = {
            'M': output_f,
            'Feature_final': final_f,
            'Feature_l': proj_x_l.mean(dim=0), 
            'Feature_a': proj_x_a.mean(dim=0), 
            'Feature_v': proj_x_v.mean(dim=0),
            'gt_l': gt_l,
            'gt_a': gt_a,
            'gt_v': gt_v,
            'loss_score': loss_score,
            'loss_ib': loss_ib,
            'loss_cond': loss_cond,
            'loss_dec': loss_dec
        }
        return res
