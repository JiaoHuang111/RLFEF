import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import grad
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)
MAX_ATOMIC_NUM=100

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from scipy.optimize import linear_sum_assignment

import pdb


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model


    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError
class RequiresGradContext(object):
    def __init__(self, *objs, requires_grad):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)


class CSPEnergy(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 初始化解码器，使用hydra工具动态实例化，传入一些参数
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.time_dim, pred_scalar = True, smooth = True)
        # 初始化beta调度器
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        # 初始化sigma调度器
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        # 时间维度
        self.time_dim = self.hparams.time_dim
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        # 检查并设置update_type属性
        if not hasattr(self.hparams, 'update_type'):
            self.update_type = True
        else:
            self.update_type = self.hparams.update_type      


    def forward(self, batch):
        # 获取batch大小
        batch_size = batch.num_graphs
        # 从beta调度器中均匀采样时间
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        # 获取时间嵌入
        time_emb = self.time_embedding(times)
        # 获取时间对应的alpha积累产物和beta
        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]
        # 计算c0和c1的值
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)
        # 获取sigma和sigma_norm
        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        # 将晶格参数转换为矩阵
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        # 获取分数坐标
        frac_coords = batch.frac_coords
        # 生成随机晶格和随机坐标
        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
        # 计算输入晶格和输入分数坐标
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.
        # 将原子类型转换为one-hot编码
        gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
        # 生成随机类型
        rand_t = torch.randn_like(gt_atom_types_onehot)
        # 根据update_type确定原子类型概率
        if self.update_type:
            atom_type_probs = (c0.repeat_interleave(batch.num_atoms)[:, None] * gt_atom_types_onehot + c1.repeat_interleave(batch.num_atoms)[:, None] * rand_t)
        else:
            atom_type_probs = gt_atom_types_onehot

        # 通过解码器预测能量
        pred_e = self.decoder(time_emb, atom_type_probs, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        # 计算能量损失
        loss_energy = F.l1_loss(pred_e, batch.y)


        loss = loss_energy

        return {
            'loss' : loss,
        }

    @torch.no_grad()
    def sample(self, batch, uncod, diff_ratio = 1.0, step_lr = 1e-5, aug = 1.0):  # diff ratio: 0.1→1
        # 获取批量大小
        batch_size = batch.num_graphs
        # 初始化(T时刻)随机晶格和随机坐标
        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        update_type = self.update_type

        # 根据update_type决定是否初始化原子种类t_T
        t_T = torch.randn([batch.num_nodes, MAX_ATOMIC_NUM]).to(self.device) if update_type else F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()

        # 根据diff_ratio进行初始化
        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
            frac_coords = batch.frac_coords
            rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)
            rand_t = torch.randn_like(atom_types_onehot)
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]
            beta = self.beta_scheduler.betas[time_start]
            c0 = torch.sqrt(alphas_cumprod)
            c1 = torch.sqrt(1. - alphas_cumprod)
            sigmas = self.sigma_scheduler.sigmas[time_start]
            l_T = c0 * lattices + c1 * rand_l
            x_T = (frac_coords + sigmas * rand_x) % 1.
            t_T = c0 * atom_types_onehot + c1 * rand_t if update_type else atom_types_onehot

        else:
            time_start = self.beta_scheduler.timesteps
        # 初始化轨迹字典:将T时刻的种类保存
        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : t_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        for t in tqdm(range(time_start, 0, -1)):
            # 在时间步t创建一个大小为batch_size的张量
            times = torch.full((batch_size, ), t, device = self.device)
            # 获取时间嵌入
            time_emb = self.time_embedding(times)
            # 如果有潜在维度，则将时间嵌入和潜在变量z拼接
            if self.hparams.latent_dim > 0:            
                time_emb = torch.cat([time_emb, z], dim = -1)
            # 根据当前时间步生成随机晶格和随机原子类型
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T)
            # 获取当前时间步的alpha、alpha积累产物和sigma值
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            # 计算c0、c1和c2的值
            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            c2 = (1 - alphas) / torch.sqrt(alphas)
            # 获取当前时间步的晶格、分数坐标和原子类型
            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']
            t_t = traj[t]['atom_types']

            # Corrector
            # 根据当前时间步生成随机晶格和随机原子类型
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # 计算步长和标准差
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)
            # 如果需要更新原子类型
            if update_type:
                # 通过解码器预测晶格、分数坐标和原子类型(todo:这里的decoder有没有用到测试集的数据？)
                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t, x_t, l_t, batch.num_atoms, batch.batch)
                # 调整预测的分数坐标
                pred_x = pred_x * torch.sqrt(sigma_norm)
                # 更新分数坐标
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_t
            
            else:
                # 获取原子类型的one-hot编码
                t_t_one = t_T.argmax(dim=-1).long() + 1
                # 通过解码器预测晶格和分数坐标
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t, l_t, batch.num_atoms, batch.batch)
                # 调整预测的分数坐标
                pred_x = pred_x * torch.sqrt(sigma_norm)
                # 更新分数坐标
                x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x
                l_t_minus_05 = l_t
                t_t_minus_05 = t_T

            # Predictor
            # 根据当前时间步生成随机晶格和随机原子类型
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_t = torch.randn_like(t_T) if t > 1 else torch.zeros_like(t_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # 获取相邻时间步的sigma值，计算步长和标准差
            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))
            # 如果需要更新原子类型
            if update_type:
                # 通过解码器预测晶格、分数坐标和原子类型
                pred_l, pred_x, pred_t = uncod.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                # 使用梯度计算能量并获取梯度值
                with torch.enable_grad():
                    with RequiresGradContext(t_t_minus_05, x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_t, grad_x, grad_l = grad(pred_e, [t_t_minus_05, x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)
                # 调整预测的分数坐标
                pred_x = pred_x * torch.sqrt(sigma_norm)

                # 更新分数坐标、晶格和原子类型
                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) - (sigmas ** 2) * aug * grad_t + sigmas * rand_t

            else:
                # 获取原子类型的one-hot编码
                t_t_one = t_T.argmax(dim=-1).long() + 1
                # 通过解码器预测晶格和分数坐标
                pred_l, pred_x = uncod.decoder(time_emb, t_t_one, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                # 使用梯度计算能量并获取梯度值
                with torch.enable_grad():
                    with RequiresGradContext(x_t_minus_05, l_t_minus_05, requires_grad=True):
                        pred_e = self.decoder(time_emb, t_T, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)
                        grad_outputs = [torch.ones_like(pred_e)]
                        grad_x, grad_l = grad(pred_e, [x_t_minus_05, l_t_minus_05], grad_outputs = grad_outputs, allow_unused=True)
                # 调整预测的分数坐标
                pred_x = pred_x * torch.sqrt(sigma_norm)
                # 更新分数坐标、晶格和原子类型
                x_t_minus_1 = x_t_minus_05 - step_size * pred_x - (std_x ** 2) * aug * grad_x + std_x * rand_x 

                l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) - (sigmas ** 2) * aug * grad_l + sigmas * rand_l 

                t_t_minus_1 = t_T

            # 将当前时间步的轨迹保存到traj字典中
            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : t_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }
        # 将轨迹堆叠
        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(time_start, -1, -1)]).argmax(dim=-1) + 1,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }
        # 获取最终的预测结果
        res = traj[0]
        res['atom_types'] = res['atom_types'].argmax(dim=-1) + 1
        # 返回最终结果和堆叠的轨迹
        return traj[0], traj_stack

    def multinomial_sample(self, t_t, pred_t, num_atoms, times):
        
        noised_atom_types = t_t
        pred_atom_probs = F.softmax(pred_t, dim = -1)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)

        return theta



    def type_loss(self, pred_atom_types, target_atom_types, noised_atom_types, batch, times):

        pred_atom_probs = F.softmax(pred_atom_types, dim = -1)

        atom_probs_0 = F.one_hot(target_atom_types - 1, num_classes=MAX_ATOMIC_NUM)

        alpha = self.beta_scheduler.alphas[times].repeat_interleave(batch.num_atoms)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times - 1].repeat_interleave(batch.num_atoms)

        theta = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * atom_probs_0 + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)
        theta_hat = (alpha[:, None] * noised_atom_types + (1 - alpha[:, None]) / MAX_ATOMIC_NUM) * (alpha_bar[:, None] * pred_atom_probs + (1 - alpha_bar[:, None]) / MAX_ATOMIC_NUM)

        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        theta_hat = theta_hat / (theta_hat.sum(dim=-1, keepdim=True) + 1e-8)

        theta_hat = torch.log(theta_hat + 1e-8)

        kldiv = F.kl_div(
            input=theta_hat, 
            target=theta, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)

        return kldiv.mean()

    def lap(self, probs, types, num_atoms):
        
        types_1 = types - 1
        atoms_end = torch.cumsum(num_atoms, dim=0)
        atoms_begin = torch.zeros_like(num_atoms)
        atoms_begin[1:] = atoms_end[:-1]
        res_types = []
        for st, ed in zip(atoms_begin, atoms_end):
            types_crys = types_1[st:ed]
            probs_crys = probs[st:ed]
            probs_crys = probs_crys[:,types_crys]
            probs_crys = F.softmax(probs_crys, dim=-1).detach().cpu().numpy()
            assignment = linear_sum_assignment(-probs_crys)[1].astype(np.int32)
            types_crys = types_crys[assignment] + 1
            res_types.append(types_crys)
        return torch.cat(res_types)



    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):

        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
        }

        return log_dict, loss

    