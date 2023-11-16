from pgmpy.estimators import PC, MmhcEstimator, HillClimbSearch, BicScore
from pyCausalFS.CBD.MBs.MMMB.MMPC import MMPC
from causallearn.search.PermutationBased.GRaSP import grasp
import argparse
import drsl
import util
import numpy as np
import time
import os
import sys


def run_algorithm(data, method_name, epsilon = 0.1, start_skel = None):

    print('--------------------{}--------------------'.format(method_name))

    if method_name in ['dro_wass', 'dro_kl', 'reg_lr']:
        return drsl.skeleton_learn(data, method_name, epsilon)
    elif method_name in ['pc']:
        # pyCausalFS, fast
        _, n_nodes = data.shape
        est_graph = np.zeros((n_nodes, n_nodes), dtype = bool)
        for target in range(n_nodes):
            pc, _, _ = MMPC(data, target, 0.01, True)
            for u in pc:
                est_graph[u, target] = True
        est_graph = est_graph + est_graph.T
        print(est_graph)
        return est_graph

        # pgmpy, PC, slow
        '''
        est_graph, _ = PC(data).estimate(variant = 'orig', return_type = 'skeleton', max_cond_vars = 2)
        est_skeleton = util.bn_to_skeleton(est_graph)
        return est_skeleton
        '''
    elif method_name in ['grasp']:
        try:
            data, _ = util.process_df_data(data)
            g = grasp(data, maxP = 3)
            est_graph = g.graph
            est_graph = est_graph.astype(bool)
            est_graph = est_graph + est_graph.T
            print(type(est_graph))
            print(est_graph)
            return est_graph
        except:
            print('An error occurred when runing GRASP.')
            _, n_nodes = data.shape
            est_graph = np.zeros((n_nodes, n_nodes), dtype = bool)
            return est_graph

    elif method_name in ['hc']:
        # bnsl stable but a lil slow
        if start_skel is None:
            est_graph = HillClimbSearch(data).estimate(scoring_method = 'bicscore', max_iter = 1e3)
        else:
            one_dag = util.skel_to_dag(start_skel, list(data.columns))
            skel_edges = util.adj_to_edges(start_skel, list(data.columns))
            est_graph = HillClimbSearch(data).estimate(scoring_method = 'bicscore', max_iter = 1e3, white_list = skel_edges, start_dag = None)
        est_dag = util.bn_to_dag(est_graph)
        
        return est_dag


    return None


def single_mode():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type = str, required = True)
    argparser.add_argument('--samples', type = int, required = True)
    argparser.add_argument('--noise', type = str, required = True, choices = ['huber', 'independent', 'noisefree'])
    argparser.add_argument('--pnoise', type = float, required = True)
    argparser.add_argument('--method', type = str, required = True, choices = ['dro_wass', 'dro_kl', 'reg_lr'])
    argparser.add_argument('--epsilon', type = float, required = True)
    argparser.add_argument('--threshold', type = float, required = True)
    args = argparser.parse_args()
    # args = argparser.parse_args('--dataset data/cancer.bif --samples 1000 --noise noisefree --pnoise 0 --method dro_wass --epsilon 1 --threshold 0.1'.split())

    dataset_path = args.dataset
    num_samples = args.samples
    noise_model = args.noise
    p_noise = args.pnoise
    method_name = args.method
    epsilon = args.epsilon
    threshold = args.threshold

    gt_model, card_dict = util.load_bn_model(dataset_path)
    gt_dag = util.bn_to_dag(gt_model)
    gt_skel = util.bn_to_skeleton(gt_model)
    print(gt_skel)
    clean_samples_df = util.sampling(gt_model, num_samples)
    noisy_samples = util.perturb_data(clean_samples_df, card_dict, noise_model, p_noise)
    est_weight_mat = run_algorithm(noisy_samples, method_name, epsilon)
    
    est_skel = util.skel_by_threshold(est_weight_mat, thr = threshold)
    print('best:')
    print(est_skel)
    print('f1:')
    print(util.f1_score(gt_skel, est_skel))

    return


def exp_mode_bif():

    t0 = time.time()

    drsl.wass_norm = 'l1'
    dataset_folder = 'data'
    # dataset_name_list = ['asia', 'cancer', 'earthquake', 'sachs', 'survey', 'alarm', 'barley', 'child', 'insurance', 'mildew', 'water']
    dataset_name_list = ['cancer']
    num_samples = 5000
    noise_model = 'noisefree'
    p_noise = 0.0
    num_exps = 1

    all_res = []
    for dataset_name in dataset_name_list:
        print('********************')
        print('DATASET: {}'.format(dataset_name))
        dataset_path = os.path.join(dataset_folder, '{}.bif'.format(dataset_name))
        gt_model, card_dict = util.load_bn_model(dataset_path)
        gt_dag = util.bn_to_dag(gt_model)
        gt_skel = util.bn_to_skeleton(gt_model)

        print('--------ground truth--------')
        print(gt_skel)
        
        res = []
        for exp_i in range(num_exps):
            clean_samples_df = util.sampling(gt_model, num_samples)
            noisy_samples = util.perturb_data(clean_samples_df, card_dict, noise_model, p_noise)

            est_skel_wass = run_algorithm(noisy_samples, 'dro_wass', epsilon = 1)
            est_skel_kl = run_algorithm(noisy_samples, 'dro_kl', epsilon = 1)
            est_skel_reg = run_algorithm(noisy_samples, 'reg_lr', epsilon = 0.01)
            est_skel_wass = util.skel_by_threshold(est_skel_wass, thr = 0.1)
            est_skel_kl = util.skel_by_threshold(est_skel_kl, thr = 0.1)
            est_skel_reg = util.skel_by_threshold(est_skel_reg, thr = 0.1)
            est_skel_pc = run_algorithm(noisy_samples, 'pc')
            est_skel_grasp = run_algorithm(noisy_samples, 'grasp')

            est_dag_wass = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_wass)
            est_dag_kl = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_kl)
            est_dag_reg = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_reg)
            est_dag_pc = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_pc)
            est_dag_grasp = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_grasp)

            est_dag_hc = run_algorithm(noisy_samples, 'hc', start_skel = None)

            if 'est_skel_wass' in locals():
                res.append(util.f1_score(gt_skel, est_skel_wass))
            if 'est_skel_kl' in locals():
                res.append(util.f1_score(gt_skel, est_skel_kl))
            if 'est_skel_reg' in locals():
                res.append(util.f1_score(gt_skel, est_skel_reg))
            if 'est_skel_pc' in locals():
                res.append(util.f1_score(gt_skel, est_skel_pc))
            if 'est_skel_grasp' in locals():
                res.append(util.f1_score(gt_skel, est_skel_grasp))
            if 'est_dag_wass' in locals():
                res.append(util.f1_score(gt_dag, est_dag_wass))
            if 'est_dag_kl' in locals():
                res.append(util.f1_score(gt_dag, est_dag_kl))
            if 'est_dag_reg' in locals():
                res.append(util.f1_score(gt_dag, est_dag_reg))
            if 'est_dag_pc' in locals():
                res.append(util.f1_score(gt_dag, est_dag_pc))
            if 'est_dag_grasp' in locals():
                res.append(util.f1_score(gt_dag, est_dag_grasp))
            if 'est_dag_hc' in locals():
                res.append(util.f1_score(gt_dag, est_dag_hc))

        res = np.array(res).reshape(num_exps, -1)
        for x in res:
            print(x)
        all_res.append(res)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('ALL RESULTS:')
    for dataset_name, res in zip(dataset_name_list, all_res):
        print(dataset_name)
        res = np.array(res).T
        for ele in res.mean(1):
            print(ele)
        for row in res:
            print('\t'.join(row.astype(str).tolist()))
    
    t1 = time.time()
    print('Total time elapsed: {}'.format(t1 - t0))

    return


def exp_mode_real():

    dataset_name = 'data/real/voting.csv'
    num_samples = 1000
    noise_model = 'noisefree'
    p_noise = 0.0
    num_exps = 1

    samples, card_dict = util.get_csv_bn_data(dataset_name)
    # samples, card_dict = util.get_csv_bn_data_new('dna', 'data/real/dna')
    
    res = []
    for exp_i in range(num_exps):
        train_samples, test_samples = util.split_dataset(samples)
        noisy_samples = util.perturb_data(train_samples, card_dict, noise_model, p_noise)

        est_skel_wass = run_algorithm(noisy_samples, 'dro_wass', epsilon = 1)
        est_skel_kl = run_algorithm(noisy_samples, 'dro_kl', epsilon = 1)
        est_skel_reg = run_algorithm(noisy_samples, 'reg_lr', epsilon = 0.01)
        est_skel_wass = util.skel_by_threshold(est_skel_wass, thr = 0.1)
        est_skel_kl = util.skel_by_threshold(est_skel_kl, thr = 0.1)
        est_skel_reg = util.skel_by_threshold(est_skel_reg, thr = 0.1)
        est_skel_pc = run_algorithm(noisy_samples, 'pc')
        est_skel_grasp = run_algorithm(noisy_samples, 'grasp')

        est_dag_wass = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_wass)
        est_dag_kl = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_kl)
        est_dag_reg = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_reg)
        est_dag_pc = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_pc)
        est_dag_grasp = run_algorithm(noisy_samples, 'hc', start_skel = est_skel_grasp)

        est_dag_hc = run_algorithm(noisy_samples, 'hc', start_skel = None)

        if 'est_dag_wass' in locals():
            res.append(util.bn_structure_score(est_dag_wass, test_samples))
        if 'est_dag_kl' in locals():
            res.append(util.bn_structure_score(est_dag_kl, test_samples))
        if 'est_dag_reg' in locals():
            res.append(util.bn_structure_score(est_dag_reg, test_samples))
        if 'est_dag_pc' in locals():
            res.append(util.bn_structure_score(est_dag_pc, test_samples))
        if 'est_dag_grasp' in locals():
            res.append(util.bn_structure_score(est_dag_grasp, test_samples))
        if 'est_dag_hc' in locals():
            res.append(util.bn_structure_score(est_dag_hc, test_samples))

    res = np.array(res).reshape(num_exps, -1)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('ALL RESULTS:')
    res = np.array(res).T
    for ele in res.mean(1):
        print(ele)
    for row in res:
        print('\t'.join(row.astype(str).tolist()))

    return


if __name__ == '__main__':

    single_mode()

    # exp_mode_bif()

    # exp_mode_real()

