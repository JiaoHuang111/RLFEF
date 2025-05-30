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

    frac_coords = []
    frac_coords_prob = []
    num_atoms = []
    atom_types = []
    # atom_types_prob = []
    lattices = []
    lattices_prob = []
    input_data_list = []
    index = 0
    for idx, batch in enumerate(loader):
        index += 1
        # if index != 2 and index != 3:
        #     continue
        # if index >= 4:
        #     break
        if torch.cuda.is_available():
            batch.cuda()
        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        batch_coords_prob = []
        batch_atom_types_prob = []
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

def generating_data(loader, diff_model, model_path, i=0):
    # 1. 生成一定量的数据
    print(f'Iteration {i}:')
    print('Generating crystal molecules using diffusion model.')
    # 72个batch，每个batch1000个分子，每个batch大约要1分钟，sample 0/1是什么意思？
    frac_coords, atom_types, lengths, angles, num_atoms, frac_coords_prob, lattices_prob = diffusion(loader, diff_model)
    print('Generation Done')
    # 2. 数据保存
    print('Start Storing Data...')
    diff_out_name = f'rl_csp_dif_{i}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords,
        'num_atoms': num_atoms,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles,
        'lattices_prob': lattices_prob,
        'frac_coords_prob': frac_coords_prob,
    }, model_path / diff_out_name)
    print('Storing Data Done.')

def main(args):
    '''
    (1)使用diffcsp生成新晶体分子Without A
    (2)统计这些分子的形成能
    (3)找到比数据集中好的分子
    (4)在数据集中找到化学式为ABX的分子，统计这些分子的形成能
    '''
    # 统计test数据集

    # 加载评估模型
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
    # 生成数据
    generating_data(loader, diff_model, model_path)
    # # 3. 读取数据
    print('Start Getting crystal list...')
    all_metrics = {}
    # 获取用于评价的模型名称
    eval_model_name = cfg.data.eval_model_name
    # 获取优化任务的文件路径
    # opt_file_path = Path(model_path / diff_out_name)
    opt_file_path = '/home/huangjiao/csp-rl/diffcsp/prop_models/mp20/rl_csp_dif_' + '0' + '.pt'
    # 获取晶体数组列表
    # crys_array_list, _  = get_crystal_array_list(opt_file_path)
    crys_array_list, _, [prob_x, prob_l] = get_crystal_array_list(opt_file_path, batch_idx=-3)
    # print(crys_array_list)
    print(len(crys_array_list))  # 9046
    print( len(prob_x), len(prob_l))
    # print(np.shape(crys_array_list))
    # print(np.shape(prob_a))
    print('Getting crystal list Done.')
    # 将晶体数组列表映射为 Crystal 对象
    # 4. 使用reward函数评估数据
    print('Start evaluation...')
    opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
    # print(np.shape(opt_crys))
    # opt_crys = []
    # for i, x in enumerate(crys_array_list):
    #     opt_crys.append(Crystal(x))
    #     print('done')
    # opt_crys = [Crystal(x) for x in crys_array_list]
    # 创建优化评价器对象
    print('point5')
    opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
    print('point6')
    # 获取优化评价指标
    # opt_metrics = opt_evaluator.get_metrics()  # todo:
    # all_metrics.update(opt_metrics)
    print('point7')
    props, valid_index = opt_evaluator.get_props()
    print('point8')
    print(props)
    pass


if __name__ == '__main__':

    print("Start!")
    parser = argparse.ArgumentParser()
    # reward 函数、评估函数所在文件夹的路径
    parser.add_argument('--model_path', default='/home/huangjiao/csp-rl/diffcsp/prop_models/mp20')
    # diffusion model所在文件夹的路径
    # parser.add_argument('--uncond_path', default='/home/huangjiao/csp-rl/outputs/hydra/singlerun/2024-07-09/mp20_Ab')
    parser.add_argument('--uncond_path', default='/home/huangjiao/csp-rl/outputs/hydra/singlerun/2024-09-12/mp20_csp')
    # 训练相关的超参数待定：rl的训练步数
    # parser.add_argument('--train_step', default=100, type=int)
    # parser.add_argument('--step_lr', default=1e-5, type=float)
    # parser.add_argument('--aug', default=50, type=float)
    # # 测试相关的超参数保留
    # parser.add_argument('--test_samples', default=100, type=int)
    # parser.add_argument('--num_candidates', default=10, type=int)
    # # label，默认energy
    # parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)




