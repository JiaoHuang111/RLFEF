from collections import Counter
import argparse
import os
import json

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
import pandas as pd

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import torch
import signal
from pymatgen.io.cif import CifWriter
import concurrent.futures

from pyxtal import pyxtal

import pickle

import sys
sys.path.append('.')

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict):
        # print('start crystal')
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict['atom_types'] = (np.argmax(self.atom_types, axis=-1) + 1)
            self.atom_types = (np.argmax(self.atom_types, axis=-1) + 1)
        # print('get_structure()')
        # self.get_cif()
        self.get_structure()
        # print('get_composition()')
        self.get_composition()
        # print('get_validity()')
        self.get_validity()
        # print('get_fingerprints')
        self.get_fingerprints()
        # print('crystal done')


    def get_cif(self, prop):
        # 获取cif文件
        print('starting getting cif!')
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'
        else:
            try:
                print('point1')
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True

                # 使用 Composition 将原子类型转换为化学式
                composition = Composition.from_dict(
                    {int(atom): count for atom, count in zip(*np.unique(self.atom_types, return_counts=True))})
                formula = composition.reduced_formula

                # 确定文件名，自动递增编号
                output_file = f"{formula}_{prop}.cif"
                print('point4')
                # 保存为 CIF 文件
                cif_writer = CifWriter(self.structure)
                cif_writer.write_file(output_file)
                print(f"CIF file saved as: {output_file}")

            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

        pass


    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        if np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value'            
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        # self.comp_valid = smact_validity(self.elems, self.comps)
        # if self.constructed:
        #     self.struct_valid = structure_validity(self.structure)
        # else:
        #     self.struct_valid = False
        # self.valid = self.comp_valid and self.struct_valid
        if np.any(np.isnan(self.frac_coords)) or np.any(np.isnan(self.lengths)) or np.any(np.isnan(self.angles)):
            print("Error: NaN values detected in structure properties.")
            self.struct_valid = False
            self.comp_valid = False
            self.valid = False
            return

            # Check component validity using smact_validity
        self.comp_valid = smact_validity(self.elems, self.comps)

        # If the structure is constructed, validate the structure
        if self.constructed:
            try:
                self.struct_valid = structure_validity(self.structure)
                if np.isnan(self.struct_valid):
                    raise ValueError("Invalid structure: NaN values found in validity check.")
            except Exception as e:
                print(f"Error in structure validity check: {e}")
                self.struct_valid = False
        else:
            self.struct_valid = False

        # Final validity is the conjunction of component and structure validity
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        # print('p1')
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        # print('p2')

        def featurize_with_timeout(structure, index):
            try:
                return CrystalNNFP.featurize(structure, index)
            except Exception as e:
                print(f"Error processing atom {index}: {e}")
                return None

        def handler(signum, frame):
            raise TimeoutError("Function call timed out")

        signal.signal(signal.SIGALRM, handler)
        site_fps = []
        try:
            # site_fps = []
            # print('p3')
            # site_fps = [CrystalNNFP.featurize(
            #     self.structure, i) for i in range(len(self.structure))]
            # print('p4')
            for i in range(len(self.structure)):
                signal.alarm(60)  # Set a 60-second alarm
                try:
                    result = featurize_with_timeout(self.structure, i)
                    if result is not None:
                        site_fps.append(result)
                finally:
                    signal.alarm(0)  # Disable the alarm
            # print('p4')
        except TimeoutError:
            print('TimeoutError: Fingerprint calculation took too long.')
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(featurize_with_timeout, self.structure, i) for i in
            #                range(len(self.structure))]
            #     # Retrieve results with a timeout
            #     for future in futures:
            #         try:
            #             result = future.result(timeout=60)  # 60 seconds timeout
            #             if result is not None:
            #                 site_fps.append(result)
            #         except concurrent.futures.TimeoutError:
            #             print('TimeoutError: Fingerprint calculation took too long for one or more tasks.')
            #             self.valid = False
            #             self.comp_fp = None
            #             self.struct_fp = None
            #             return
            #         except Exception as e:
            #             print(f'Error retrieving result: {e}')
            #             self.valid = False
            #             self.comp_fp = None
            #             self.struct_fp = None
            #             return
            # print('p4')
        # except concurrent.futures.TimeoutError:
        #     print('TimeoutError: Fingerprint calculation took too long.')
        #     self.valid = False
        #     self.comp_fp = None
        #     self.struct_fp = None
        #     return

        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            # print('p5')
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics



class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            with torch.no_grad():
                pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
                gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        with torch.no_grad():
            (cov_metrics_dict, combined_dist_dict) = compute_cov(
                self.crys, self.gt_crys,
                struc_cutoff=cutoff_dict['struc'],
                comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_coverage())
        return metrics

class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        # 获取有效晶体的索引，并将其转换为 numpy 数组（todo:这里的有效是如何确定的？）
        valid_indices = np.array([c.valid for c in self.crys])
        # 将有效索引重塑为(step_opt, num_opt)的形状 todo:ValueError: cannot reshape array of size 9046 into shape (90,100)
        indices_shapae = np.shape(valid_indices)
        print(indices_shapae)
        # valid_indices = valid_indices[:self.step_opt * self.num_opt]
        # valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        # 获取有效索引的行和列
        # valid_x, valid_y = valid_indices.nonzero()
        valid_x = valid_indices.nonzero()
        # 初始化属性数组，初始值为正无穷大
        props = np.ones(indices_shapae) * np.inf
        # 获取所有有效的晶体
        valid_crys = [c for c in self.crys if c.valid]
        # 如果没有有效的晶体，成功率为0
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            # 使用评估模型计算有效晶体的预测属性
            with torch.no_grad():
                pred_props = prop_model_eval(self.eval_model_name, [
                                             c.dict for c in valid_crys])
            props[valid_x] = pred_props
            props_mean = np.mean(props[props != np.inf])  # Exclude infinite values
            props_min = np.min(props[props != np.inf])  # Exclude infinite values

        return {
            # 'SR5': sr_5,
            # 'SR10': sr_10,
            # 'SR15': sr_15,
            'PropsMean': props_mean,
            'PropsMin': props_min
            # 'x_0_mean': x0_mean,
            # 'x_0_min': x0_min,
            # 'x_1_mean': x1_mean,
            # 'x_1_min': x1_min,
            # 'y_0_mean': y0_mean,
            # 'y_0_min': y0_min,
            # 'y_1_mean': y1_mean,
            # 'y_1_min': y1_min
        }


    def get_props(self):

        valid_crys = [c for  c in self.crys if c.valid]
        valid_indices = [i for i, c in enumerate(self.crys) if c.valid]
        with torch.no_grad():
            pred_props, E_hull = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])

            props_tensor = torch.tensor(pred_props, requires_grad=True)
            E_hull_tensor = torch.tensor(E_hull, requires_grad=True)
        return props_tensor, E_hull_tensor, valid_indices

    def get_SUN(self):

        valid_crys = [c for  c in self.crys if c.valid]
        valid_indices = [i for i, c in enumerate(self.crys) if c.valid]
        with torch.no_grad():
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in valid_crys])
            props_tensor = torch.tensor(pred_props, requires_grad=True)
        return props_tensor, valid_indices

    def get_metrics(self):
        return self.get_success_rate()


def get_file_paths(root_path, task, label='', suffix='pt'):
    if args.label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'])
    elif batch_idx == -3:
        crys_array_list, probs = get_crystals_list(
            data['frac_coords'][0],
            data['atom_types'][0],
            data['lengths'][0],
            data['angles'][0],
            data['num_atoms'][0],
            data['frac_coords_prob'][0], data['lattices_prob'][0])
    else:
        crys_array_list, probs = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx],
        data['atom_types_prob'][batch_idx], data['frac_coords_prob'][batch_idx], data['lattices_prob'][batch_idx])
        #probs = [data['atom_types_prob'][batch_idx], data['frac_coords_prob'][batch_idx], data['lattices_prob'][batch_idx]]

    if 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list, probs


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles)
    }
    return Crystal(crys_array_dict) 

def main(args):
    # 初始化一个空的字典，用于存储所有的评价指标
    all_metrics = {}
    # 加载配置文件
    cfg = load_config(args.root_path)
    # 获取用于评价的模型名称
    eval_model_name = cfg.data.eval_model_name

    if 'opt' in args.tasks:
        # 获取优化任务的文件路径
        opt_file_path = get_file_paths(args.root_path, 'opt', args.label)
        # 获取晶体数组列表
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        # 将晶体数组列表映射为 Crystal 对象
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
        # 创建优化评价器对象
        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        # 获取优化评价指标
        opt_metrics = opt_evaluator.get_metrics()
        # 将优化评价指标更新到所有评价指标字典中
        all_metrics.update(opt_metrics)
    elif 'go' in args.tasks:
        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        crys_array_list, _, = get_crystal_array_list(gen_file_path, batch_idx=-2)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)
        # 创建优化评价器对象
        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        # 获取优化评价指标
        opt_metrics = opt_evaluator.get_metrics()
        # 将优化评价指标更新到所有评价指标字典中
        all_metrics.update(opt_metrics)
    elif 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
        crys_array_list, _, = get_crystal_array_list(gen_file_path, batch_idx = -2)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)


    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))   

        if args.multi_eval:
            rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
        else:
            rec_evaluator = RecEval(pred_crys, gt_crys)

        recon_metrics = rec_evaluator.get_metrics()

        all_metrics.update(recon_metrics)

    # 打印所有的评价指标
    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_metrics.json'
    else:
        metrics_out_file = f'eval_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    args = parser.parse_args()
    main(args)
