from scipy.optimize import minimize, LinearConstraint, check_grad, Bounds
import numpy as np
import itertools
import torch
import stochastic
import util

wass_norm = 'l1'

def get_penalty_matrix(n):

    if wass_norm == 'l1':
        ret = (1 - np.eye(n)) * 2
        ret[:-1, -1] = n
        ret[-1, :-1] = n
    elif wass_norm == 'l2':
        ret = (1 - np.eye(n)) * np.sqrt(2)
        ret[:-1, -1] = np.sqrt(n + 2)
        ret[-1, :-1] = np.sqrt(n + 2)
    elif wass_norm == 'linf':
        ret = (1 - np.eye(n))
        ret[:-1, -1] = 2
        ret[-1, :-1] = 2
    elif wass_norm == 'ham':
        ret = (1 - np.eye(n))
    else:
        pass

    return ret


def get_random_permutation(n, start_num = -1):

    if start_num >= 0:
        ret = np.roll(np.arange(n), n - start_num)
        np.random.shuffle(ret[1:])
    else:
        ret = np.random.permutation(n)

    return ret


def get_encoding_matrix(n):
    
    ret = np.eye(n, n - 1)
    ret[-1] = -1
    
    return ret


def get_block_norm_vec(block_mat, l_ind, r_ind):

    n_nodes = l_ind.shape[0]
    ret = np.zeros(n_nodes)
    for i in range(n_nodes):
        ret[i] = np.linalg.norm(block_mat[l_ind[i] : r_ind[i], :])

    return ret


def encode_data(input_data, cards):

    n_data, n_nodes = input_data.shape
    n_enc_dim = (cards - 1).sum()
    output_data = np.zeros((n_data, n_enc_dim))
    
    l_ind = np.zeros(n_nodes, dtype = int)
    r_ind = np.zeros(n_nodes, dtype = int)
    for i in range(1, n_nodes):
        l_ind[i] = l_ind[i - 1] + cards[i - 1] - 1
        r_ind[i - 1] = l_ind[i]
    r_ind[-1] = n_enc_dim
    
    for i in range(n_nodes):
        enc_mat = get_encoding_matrix(cards[i])
        output_data[:, l_ind[i] : r_ind[i]] = enc_mat[input_data[:, i]]

    return output_data, l_ind, r_ind


def dro_wass_worst_dist(W, gamma, cur_node, data, data_ori, l_ind, r_ind, cards, epsilon, max_iter, exact):

    n_data, n_enc_dim = data.shape
    n_nodes = cards.size
    n_enc_cof = cards.sum()
    l1_ind = l_ind + np.arange(n_nodes)
    r1_ind = r_ind + np.arange(1, n_nodes + 1)

    WW = torch.zeros((n_enc_cof, cards[cur_node] - 1))
    data_pen = torch.zeros((n_data, n_enc_cof))
    for i in range(n_nodes):
        enc_mat = torch.tensor(get_encoding_matrix(cards[i]))
        pen_mat = torch.tensor(get_penalty_matrix(cards[i]))
        WW[l1_ind[i] : r1_ind[i], :] = enc_mat.matmul(W[l_ind[i] : r_ind[i], :])
        data_pen[:, l1_ind[i] : r1_ind[i]] = pen_mat[data_ori[:, i]]

    WW = torch.einsum('ij,kj->ik', WW, WW)
    WW[torch.eye(n_enc_cof, dtype = bool)] *= 0.5
    WW_diag = WW.diagonal()
    data_pen *= -gamma

    best_sel = torch.zeros((n_data, n_nodes), dtype = int)
    best_val = -torch.inf * torch.ones(n_data)
    for try_i in range(max_iter):
        node_order = torch.tensor(np.random.permutation(n_nodes))
        start_node = node_order[0]
        start_idx = np.random.randint(l1_ind[start_node], r1_ind[start_node])
        cur_sel = torch.zeros((n_data, n_nodes), dtype = int)
        cur_sel[:, 0] = start_idx
        cur_val = WW[start_idx, start_idx] + data_pen[:, start_idx]
        for i in range(1, n_nodes):
            node_i = node_order[i]
            seg_i = torch.arange(l1_ind[node_i], r1_ind[node_i])
            sub_mat = WW[seg_i[:, None, None], cur_sel[None, :, :i]].sum(-1).T + WW_diag[None, seg_i] + data_pen[:, seg_i]
            max_indices = sub_mat.argmax(1)
            max_vals = sub_mat[torch.arange(n_data), max_indices]
            cur_sel[:, i] = max_indices + l1_ind[node_i]
            cur_val += max_vals
        update_flag = (cur_val > best_val)
        best_sel[update_flag] = cur_sel[update_flag]
        best_val[update_flag] = cur_val[update_flag]

    one_hot = torch.zeros((n_data, n_enc_cof)).scatter_(1, best_sel, 1).float()
    ret = torch.zeros(data.shape)
    for i in range(n_nodes):
        enc_mat = torch.tensor(get_encoding_matrix(cards[i]), dtype = torch.float32)
        ret[:, l_ind[i] : r_ind[i]] = one_hot[:, l1_ind[i] : r1_ind[i]].matmul(enc_mat)

    # ret = ret.numpy()

    # greedy_val = np.square(ret.dot(W)).sum(1) / 2 - gamma * np.abs(ret- data).sum(1)
    # print(greedy_val)
    
    # exact solution
    if exact:
        best_val = -torch.inf * torch.ones(n_data)
        ret = torch.zeros((n_data, n_enc_dim))
        for val_tup in itertools.product(*[np.arange(ele) for ele in cards]):
            tem_data = torch.zeros((n_data, n_enc_dim))
            for i in range(n_nodes):
                enc_mat = torch.tensor(get_encoding_matrix(cards[i]))
                tem_data[:, l_ind[i] : r_ind[i]] = enc_mat[val_tup[i]]
            cur_val = torch.square(torch.mm(tem_data, W.float())).sum(1) / 2
            cur_val = cur_val - gamma * torch.abs(tem_data - data).sum(1)
            mask = best_val < cur_val
            best_val = torch.maximum(best_val, cur_val)
            ret[mask] = tem_data[mask]

    return ret


def dro_wass_func_grad(params, cur_node, data, data_ori, l_ind, r_ind, cards, epsilon, batch_size = 500, worst_dist_max_iter = 10, exact = False):
    
    batch = get_random_permutation(data.shape[0])[:batch_size]
    data = data[batch]
    data_ori = data_ori[batch]

    n_data, n_enc_dim = data.shape
    n_nodes = cards.size

    epsilon /= n_data

    params = torch.tensor(params)
    W = params[:-1].reshape(n_enc_dim, cards[cur_node] - 1)
    gamma = params[-1]
    W[l_ind[cur_node] : r_ind[cur_node]] = -torch.eye(cards[cur_node] - 1)

    data_Q = dro_wass_worst_dist(W, gamma, cur_node, data, data_ori, l_ind, r_ind, cards, epsilon, max_iter = worst_dist_max_iter, exact = exact)
    data_Q = data_Q.double()

    grad_W = torch.einsum('ij,ik,kl->jl', data_Q, data_Q, W) / n_data
    
    if wass_norm == 'l1':
        norm_vec = torch.linalg.vector_norm(data_Q - data, ord = 1, dim = 1)
    elif wass_norm == 'l2':
        norm_vec = torch.linalg.vector_norm(data_Q - data, ord = 2, dim = 1)
    elif wass_norm == 'linf':
        norm_vec = torch.linalg.vector_norm(data_Q - data, ord = np.inf, dim = 1)
    elif wass_norm == 'ham':
        norm_vec = torch.all(data_Q - data < 1e-6, dim = 1).float()

    grad_gamma = epsilon - norm_vec.sum() / n_data
    obj_val = grad_gamma * gamma + (grad_W * W).sum() / 2
    grad_W[l_ind[cur_node] : r_ind[cur_node]] = 0

    obj_val = obj_val.item()
    grads = torch.cat((grad_W.reshape(-1), torch.tensor([grad_gamma]))).numpy()


    return obj_val, grads


def dro_kl_func_grad(params, cur_node, data, data_ori, l_ind, r_ind, cards, epsilon):

    n_data, n_enc_dim = data.shape
    n_nodes = cards.size

    epsilon /= n_data

    params = torch.tensor(params)
    W = params[:-1].reshape(n_enc_dim, cards[cur_node] - 1)
    gamma = params[-1]
    W[l_ind[cur_node] : r_ind[cur_node]] = -torch.eye(cards[cur_node] - 1)
    data = torch.tensor(data)

    grad_vec = torch.einsum('ij,ik,kl->ijl', data, data, W)
    loss_vec = (grad_vec * W[None]).sum((1, 2)) / gamma / 2

    soft_prob = util.soft_max(loss_vec)
    log_mean_loss = util.log_exp_sum(loss_vec) - torch.tensor(n_data).log()
    obj_val = gamma * log_mean_loss + gamma * epsilon
    grad_W = (soft_prob.reshape(n_data, 1, 1) * grad_vec).sum(0)
    grad_W[l_ind[cur_node] : r_ind[cur_node]] = 0
    grad_gamma = log_mean_loss - soft_prob.dot(loss_vec) + epsilon

    obj_val = obj_val.item()
    grads = torch.cat((grad_W.reshape(-1), torch.tensor([grad_gamma]))).numpy()

    return obj_val, grads


def reg_lr_func_grad(params, cur_node, data, data_ori, l_ind, r_ind, cards, lambd):

    n_data, n_enc_dim = data.shape
    n_nodes = cards.size

    params = np.array(params)
    W = params[:-1].reshape(n_enc_dim, cards[cur_node] - 1)
    W[l_ind[cur_node] : r_ind[cur_node]] = -np.eye(cards[cur_node] - 1)
    norm_vec_W = get_block_norm_vec(W, l_ind, r_ind)

    obj_val = np.square(data.dot(W)).sum() / (2 * n_data) + lambd * norm_vec_W.sum()
    grad_W = data.T.dot(data).dot(W) / n_data

    grad_norm = np.array(W)
    for i in range(n_nodes):
        grad_norm[l_ind[i] : r_ind[i]] /= norm_vec_W[i]
    grad_W += lambd * grad_norm
    grad_W[l_ind[cur_node] : r_ind[cur_node]] = 0

    grads = np.concatenate((grad_W.reshape(-1), np.zeros(1)))

    return obj_val, grads


def skeleton_learn(data_df, method_name, epsilon):

    opt_func_dict = {'dro_wass': dro_wass_func_grad,
                     'dro_kl': dro_kl_func_grad,
                     'reg_lr': reg_lr_func_grad}
    
    n_data, n_nodes = data_df.shape
    weight_mat = np.zeros((n_nodes, n_nodes))

    data_np, cards = util.process_df_data(data_df)
    data, l_ind, r_ind = encode_data(data_np, cards)
    n_enc_dim = (cards - 1).sum()
    
    opt_func_grad = opt_func_dict[method_name]

    for cur_node in range(n_nodes):
        print('node {}'.format(cur_node))
        init_params = np.random.rand(n_enc_dim * (cards[cur_node] - 1) + 1)
        args_tup = (cur_node, data, data_np, l_ind, r_ind, cards, epsilon)
        func_grad_x = lambda x : opt_func_grad(x, *list(args_tup))
        if method_name in ['dro_wass']:
            opt_params = stochastic.adam(func_grad_x, init_params, learning_rate = 1, maxiter = 200, use_proj = True, verbose = False)
        else:
            opt_params = minimize(func_grad_x, init_params, method = 'L-BFGS-B', jac = True, bounds = [(None, None)] * (init_params.size - 1) + [(1e-9, None)], options = {'disp' : 0, 'maxiter' : 100})
        opt_params = np.array(opt_params.x)
        opt_W = opt_params[:-1].reshape(n_enc_dim, cards[cur_node] - 1)
        opt_gamma = opt_params[-1]
        opt_W[l_ind[cur_node] : r_ind[cur_node]] = 0
        norm_vec = get_block_norm_vec(opt_W, l_ind, r_ind)
        weight_mat[:, cur_node] = norm_vec

    weight_mat *= (1 - np.eye(n_nodes))
    np.set_printoptions(precision = 4, floatmode = 'fixed', suppress = True)
    
    print('weight mat:')
    print(weight_mat)

    return weight_mat

