import argparse
from hydra import initialize_config_dir
from hydra.experimental import compose
import hydra
import numpy as np
import os
import torch
import torch.optim as optim
from pathlib import Path
from torch_geometric.data import Batch
from p_tqdm import p_map
from compute_metrics import Crystal, get_crystal_array_list, OptEval
from eval_utils import lattices_to_params_shape
from collections import Counter, defaultdict
import pickle
from datetime import datetime
import pandas as pd


def load_model(model_path, load_data=False, testing=True):
    with initialize_config_dir(str(model_path)):
        # 加载配置文件
        cfg = compose(config_name='hparams')
        # 使用配置文件实例化模型
        print('loading model...')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        # 获取模型路径中的所有ckpt文件
        print('Getting ckpt...')
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt = None
            # 查找包含 'last' 的ckpt文件
            for ck in ckpts:
                if 'last' in ck.parts[-1]:
                    ckpt = str(ck)
            # 如果没有 'last' 的ckpt文件，则选择最新的一个
            if ckpt is None:
                ckpt_epochs = np.array(
                    [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts if 'last' not in ckpt.parts[-1]])
                ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        # 加载hparams配置文件
        hparams = os.path.join(model_path, "hparams.yaml")
        # 从检查点加载模型
        print('loading ckpt...')
        model = model.load_from_checkpoint(ckpt, hparams_file=hparams, strict=False)
        try:
            model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
            model.scaler = torch.load(model_path / 'prop_scaler.pt')
        except:
            pass
        print('loading data...')
        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                train_loader = datamodule.train_dataloader(shuffle=False)
                val_loader = datamodule.val_dataloader()[0]
                test_loader = (train_loader, val_loader)
        else:
            test_loader = None
    return model, test_loader, cfg


def diffusion(loader, model, num_evals = 1, step_lr = 1e-5):
    """
    每个batch128个graph，
    """
    frac_coords = []
    frac_coords_prob = []
    num_atoms = []
    atom_types = []
    # atom_types_prob = []
    lattices = []
    lattices_prob = []
    input_data_list = []
    # index = 0
    for idx, batch in enumerate(loader):
        # index += 1
        # if index != 2 and index != 3:
        #     continue
        if idx < 0:
            continue
        if idx >= 10:
            break
        if torch.cuda.is_available():
            batch.cuda()
        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        batch_coords_prob = []
        # batch_atom_types_prob = []
        batch_lattices_prob = []
        for eval_idx in range(num_evals):

            print(f'batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr = step_lr)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())
            # batch_atom_types_prob.append(outputs['atom_types_prob'].detach().cpu())
            batch_coords_prob.append(outputs['frac_coords_prob'].detach().cpu())
            batch_lattices_prob.append(outputs['lattices_prob'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))
        frac_coords_prob.append(torch.stack(batch_coords_prob, dim=0))
        # atom_types_prob.append(torch.stack(batch_atom_types_prob, dim=0))
        lattices_prob.append(torch.stack(batch_lattices_prob, dim=0))
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    frac_coords_prob = torch.cat(frac_coords_prob, dim=1)
    # atom_types_prob = torch.cat(atom_types_prob, dim=1)
    lattices_prob = torch.cat(lattices_prob, dim=1)
    print("frac_coords shape:", frac_coords.shape)
    print("num_atoms shape:", num_atoms.shape)
    print("atom_types shape:", atom_types.shape)
    print("lattices shape:", lattices.shape)
    print("frac_coords_prob shape:", frac_coords_prob.shape)
    # print("atom_types_prob shape:", atom_types_prob.shape)
    print("lattices_prob shape:", lattices_prob.shape)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, atom_types, lengths, angles, num_atoms, frac_coords_prob, lattices_prob
    )


def generating_data(i, loader, diff_model, model_path):
    # 1. 生成一定量的数据
    print(f'Iteration {i}:')
    print('Generating crystal molecules using diffusion model.')
    # 72个batch，每个batch1000个分子，每个batch大约要1分钟，sample 0/1是什么意思？
    frac_coords, atom_types, lengths, angles, num_atoms, frac_coords_prob, lattices_prob = diffusion(loader, diff_model)
    print('Generation Done')
    # 2. 数据保存
    print('Start Storing Data...')
    if args.label == '':
        diff_out_name = f'rl06_dif_{i}.pt'
    else:
        diff_out_name = f'rlv_diff_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        # 'atom_types_prob': atom_types_prob,
        'lattices_prob': lattices_prob,
        'frac_coords_prob': frac_coords_prob,
    }, model_path / diff_out_name)
    print('Storing Data Done.')

    unique_atom_types_per_graph = []
    start_index = 0
    for num in num_atoms[0]:
        end_index = int(start_index + num.item())
        types_in_graph = atom_types[0][start_index:end_index]  # 获取当前图的原子类型
        unique_atom_types = len(set(types_in_graph.tolist()))  # 统计唯一原子类型
        unique_atom_types_per_graph.append(unique_atom_types)
        start_index = end_index  # 更新开始索引
    return unique_atom_types_per_graph


def reward_function(A,X,L):
    """
    输入t_0时刻的axl，输出reward
    """
    pass


def comput_convex_hull():
    # 定义路径
    cif_folder = "all_cifs_FE"
    data_folder = "data/mp_20"

    # 1. 统计all_cifs_FE下的CIF文件数量
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
    total_crystals = len(cif_files)
    print(f"Total crystals in all_cifs_FE: {total_crystals}")

    # 2. 读取CSV文件并构建化学式到能量数据的映射
    energy_data = {}

    for csv_file in ["train.csv", "val.csv", "test.csv"]:
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            formula = row["pretty_formula"]
            formation_energy = row["formation_energy_per_atom"]
            e_above_hull = row["e_above_hull"]

            # 计算convex hull能量
            convex_hull_energy = formation_energy - e_above_hull

            energy_data[formula] = {
                "convex_hull_energy": convex_hull_energy
            }
    return energy_data


def loss_function(A, X, L, alpha, beta, reward, A_pre, X_pre, L_pre):
    """
    输入T个时刻的AXL，输出loss
    A,X,L：
    """
    reward_loss = -alpha * reward

    kl_loss_A = 0.0
    kl_loss_X = 0.0
    kl_loss_L = 0.0
    diff_time_step = 1000
    for t in range(1, diff_time_step):
        # p_theta_result = p_theta(A[t-1], X[t-1], L[t-1])
        # p_pre_result = p_pre(A_pre[t-1], X[t-1], L[t-1])
        kl_loss_A += torch.nn.functional_kl_div(A[t], A_pre[t])
        kl_loss_X += torch.nn.functional_kl_div(X[t], X_pre[t])
        kl_loss_L += torch.nn.functional_kl_div(L[t], L_pre[t])
    loss_A = reward_loss + beta*kl_loss_A
    loss_X = reward_loss + beta*kl_loss_X
    loss_L = reward_loss + beta*kl_loss_L
    return loss_A, loss_X, loss_L


def main(args):
    # 加载reward model
    print('Loading reward model.')
    model_path = Path(args.model_path)
    with torch.no_grad():
        reward_model, _, cfg = load_model(
            model_path, load_data=False)
    # 加载diffusion model
    print('Loading diffusion model.')
    diff_path = Path(args.uncond_path)
    with torch.no_grad():
        diff_model, loader, cfg = load_model(
            diff_path, load_data=True)
    print('Model to cuda')
    if torch.cuda.is_available():
        reward_model.to('cuda')
        diff_model.to('cuda')
    optimizer = optim.Adam(diff_model.parameters(), lr=0.001)
    mean_loss_epoch = {}
    lowest_loss_epoch = {}
    all_atom_type_stats = {}
    # atom_type_stats = defaultdict(lambda: {'mean': 0, 'min': float('inf'), 'count': 0})

    # rl训练过程
    best_loss = float('inf')
    best_model_path = 'best_csp_rl.pth'

    save_interval = 50  # 每 50 轮保存一次结果到文件
    for i in range(1, args.train_step+1):
        # 统计每种节点类型数量的字典
        atom_type_stats = {}
        # 1,2 生成数据
        unique_atoms_type = generating_data(i, loader, diff_model, model_path)
        # # 3. 读取数据
        print('Start Getting crystal list...')
        all_metrics = {}
        # 获取用于评价的模型名称
        eval_model_name = cfg.data.eval_model_name
        # 获取优化任务的文件路径
        opt_file_path = '/home/huangjiao/csp-rl/diffcsp/prop_models/mp20/rl06_dif_'+str(i)+'.pt'
        # 获取晶体数组列表
        crys_array_list, _, [prob_x, prob_l] = get_crystal_array_list(opt_file_path, -3)
        # print(crys_array_list)
        # print(len(crys_array_list))  # 9046
        # print(len(prob_x), len(prob_l))
        print('Getting crystal list Done.')
        # 将晶体数组列表映射为 Crystal 对象
        # 4. 使用reward函数评估数据
        print('Start evaluation...')
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
        # 创建优化评价器对象
        # print('point5')
        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        # print('point6')
        # 获取优化评价指标
        try:
            print('point7')
            props, E_hull, valid_index = opt_evaluator.get_props()
        except:
            continue
        print('point8')
        tem_i = 0
        for tem_c in opt_crys:
            if tem_c.valid:
                tem_c.get_cif(props[tem_i])
                tem_i += 1
        # return 0
        # prob_a = prob_a[valid_index]
        # print(valid_index)
        # print(len(props))  # 7449
        # print(len(prob_x), len(prob_l))
        # print(len(valid_index))
        # print(np.shape(props))
        # print(len(prob_x[0]))
        valid_index_tensor = torch.tensor(valid_index)
        props_expanded = props.unsqueeze(1)
        prob_l = torch.tensor(prob_l).view(-1, 9)

        nan_mask = torch.isnan(props)
        inf_mask = torch.isinf(props)
        gt_10_mask = props > 10
        lt_minus_100_mask = props < -100

        # 计算有效值的掩码
        valid_mask = ~(nan_mask | inf_mask | gt_10_mask | lt_minus_100_mask)

        # 获取有效值的下标
        valid_indices = torch.nonzero(valid_mask).squeeze()
        valid_prob_x = torch.tensor(prob_x)[valid_index_tensor, :][valid_indices, :]
        valid_prob_l = torch.tensor(prob_l)[valid_index_tensor, :][valid_indices, :]
        valid_props_expanded = props_expanded[valid_indices, :] - E_hull[valid_indices, :]
        valid_unique_atoms_type_num = torch.tensor(unique_atoms_type)[valid_index_tensor][valid_indices]

        loss_X = valid_props_expanded * valid_prob_x
        loss_L = valid_props_expanded * valid_prob_l

        # print(prob_a.shape)
        # loss_A = props_expanded * torch.tensor(prob_a)[valid_index_tensor, :]
        # loss_X = props_expanded * torch.tensor(prob_x)[valid_index_tensor, :]
        # loss_L = props_expanded * torch.tensor(prob_l)[valid_index_tensor, :]
        # all_loss = loss_A + loss_X + loss_L
        print('all_metrics:', props)
        # print('all_metrics keys:', list(all_metrics.keys()))
        # for key in all_metrics:
        #     print(f'shape of all_metrics[{key}]:', np.shape(all_metrics[key]))
        print('Evaluation Done.')
        # 5. 基于reward更新网络
        print('Start updating...')
        batch_size = 32
        num_batches = len(loss_X) // batch_size
        # batches_A = torch.split(loss_A, batch_size)
        # for j, batch in enumerate(batches_A):
        #     optimizer.zero_grad()  # 清空梯度
        #     batch_loss = batch.mean()
        #     batch_loss.backward(retain_graph=True)
        #     optimizer.step()
        #     print(f'Batch {j + 1}/{num_batches} updated')
        batches_X = torch.split(loss_X, batch_size)
        for j, batch in enumerate(batches_X):
            optimizer.zero_grad()  # 清空梯度
            batch_loss = batch.mean()
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            print(f'Batch {j + 1}/{num_batches} updated')
        batches_L = torch.split(loss_L, batch_size)
        for j, batch in enumerate(batches_L):
            optimizer.zero_grad()  # 清空梯度
            batch_loss = batch.mean()
            batch_loss.backward(retain_graph=True)
            optimizer.step()
            print(f'Batch {j + 1}/{num_batches} updated')
        # optimizer.zero_grad()  # 清空梯度
        # valid_props_expanded = valid_props_expanded - 6.0
        if len(valid_props_expanded) > 0:
            # 计算有效值的平均值
            valid_mean = valid_props_expanded.mean()
            valid_min = valid_props_expanded.min()
            print(f"Valid mean of props: {valid_mean.item()}")
            print(f"Valid min of props: {valid_min.item()}")
            # **更新对应节点类型数量的字典**
            for atom_type, prop_value in zip(valid_unique_atoms_type_num.tolist(),
                                             valid_props_expanded.squeeze().tolist()):
                if atom_type not in atom_type_stats:
                    atom_type_stats[atom_type] = {'mean': 0, 'min': float('inf'), 'count': 0}
                atom_type_stats[atom_type]['mean'] += prop_value  # 累加总和
                atom_type_stats[atom_type]['count'] += 1  # 计数

                # **更新最小损失**
                if prop_value < atom_type_stats[atom_type]['min']:
                    atom_type_stats[atom_type]['min'] = prop_value
                    # **在这里计算均值**
            for atom_type in atom_type_stats:
                atom_type_stats[atom_type]['mean'] /= atom_type_stats[atom_type]['count']  # **计算均值**

            all_atom_type_stats[i] = atom_type_stats

            for atom_type in atom_type_stats:
                atom_type_stats[atom_type]['count'] = 0

        else:
            print("No valid values in props.")

        # Mean_loss_epoch.append(valid_mean)
        # Lowest_loss_epoch.append(valid_min)
        mean_loss_epoch[i] = valid_mean.item()
        lowest_loss_epoch[i] = valid_min.item()
        current_loss = valid_mean.item() # 使用 props.mean() 作为评价指标
        # 检查是否为目前最佳模型
        if current_loss < best_loss:
            best_loss = current_loss
            print(f'New best model found at step {i} with loss {best_loss}')
            # 保存模型状态
            torch.save({
                'model_state_dict': diff_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': i,
                'best_loss': best_loss,
            }, best_model_path)
        print('Update Done...')
        print('Mean_loss_epoch:')
        print(mean_loss_epoch)
        print('Lowest_loss_epoch:')
        print(lowest_loss_epoch)
        unique_counts = Counter(valid_unique_atoms_type_num)
        print('Frequency of unique atom types per graph:', unique_counts)
        # **打印每种节点类型的平均损失和最小损失**
        print(f"Round {i} atom_type_stats:", all_atom_type_stats[i])
        # print('Atom type statistics:')
        # for atom_type, stats in atom_type_stats.items():
        #     mean_loss = stats['mean'] / stats['count'] if stats['count'] > 0 else 0
        #     min_loss = stats['min']
        #     print(f"Atom type {atom_type}: Mean loss: {mean_loss}, Min loss: {min_loss}")
        # 每 50 轮保存一次
        if i % save_interval == 0:
            # 获取当前日期，并格式化为 "241030" 形式
            date_str = datetime.now().strftime("%y%m%d")
            save_path = f"training_results_{date_str}_step_{i}.txt"

            # 将当前轮数的数据保存到文件中
            with open(save_path, "wb") as f:
                pickle.dump({
                    "mean_loss_epoch": mean_loss_epoch,
                    "lowest_loss_epoch": lowest_loss_epoch,
                    "all_atom_type_stats": all_atom_type_stats
                }, f)

            print(f"Training results saved at step {i} to {save_path}")
    print('Mean_loss_epoch:')
    print(mean_loss_epoch)
    print('Lowest_loss_epoch:')
    print(lowest_loss_epoch)
    print("Total atom_type_stats:", all_atom_type_stats)


if __name__ == '__main__':
    print("Start!")
    parser = argparse.ArgumentParser()
    # reward 函数、评估函数所在文件夹的路径
    parser.add_argument('--model_path', default='/home/huangjiao/csp-rl/diffcsp/prop_models/mp20')
    # diffusion model所在文件夹的路径
    #parser.add_argument('--uncond_path', default='/home/huangjiao/csp-rl/outputs/hydra/singlerun/2024-07-09/mp20_Ab')
    parser.add_argument('--uncond_path', default='/home/huangjiao/csp-rl/outputs/hydra/singlerun/2024-10-21/mp20_csp')
    # 训练相关的超参数待定：rl的训练步数
    parser.add_argument('--train_step', default=200, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--aug', default=50, type=float)
    # 测试相关的超参数保留
    parser.add_argument('--test_samples', default=100, type=int)
    parser.add_argument('--num_candidates', default=10, type=int)
    # label，默认energy
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)