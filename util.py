from pgmpy.models import BayesianNetwork
from pgmpy.base import PDAG, DAG
from pgmpy.metrics import structure_score, log_likelihood_score
import numpy as np
import bnlearn
import csv
import pandas as pd
import os


def skel_by_threshold(skel, thr = 0.1):

    skel = (skel > thr)
    skel = skel + skel.T

    return skel


def random_simplex(k):

    a = -np.log(np.random.rand(k))
    a /= a.sum()

    return a


# log(sum_x exp(x))
def log_exp_sum(x):

    return x.max() + (x - x.max()).exp().sum().log()


def soft_max(x):

    x = (x - x.max()).exp()

    return x / x.sum()


def f1_score(gt, pred):

    overlap = (gt * pred).sum()
    f1 = 2 * overlap / (pred.sum() + gt.sum())

    return f1


def bn_structure_score(adj, data):

    dag = adj_to_dag(adj, list(data.columns))
    
    return structure_score(dag, data, scoring_method = 'bic')


def load_bn_model(dataset_name):

    model = BayesianNetwork.load(dataset_name)
    cardinality_dict = model.get_cardinality()

    return model, cardinality_dict


def adj_to_dag(adj, node_list):

    dag = DAG()
    dag.add_nodes_from(node_list)
    for u, v in zip(*list(np.where(adj))):
        dag.add_edge(node_list[u], node_list[v])

    return dag


def skel_to_dag(skel, node_list):

    return adj_to_dag(np.triu(skel, 1), node_list)


def adj_to_edges(adj, node_list):

    return [(node_list[u], node_list[v]) for u, v in zip(*list(np.where(adj)))]


def bn_to_dag(model):

    return bnlearn.dag2adjmat(model).to_numpy(bool)


def bn_to_skeleton(model):

    dag = bn_to_dag(model)
    skel = dag + dag.T

    return skel


def sampling(model, n_samples):

    if n_samples == 0:
        samples = pd.DataFrame(np.zeros((0, len(model.nodes))), columns = model.nodes)
    else:
        samples = model.simulate(n_samples)[list(model.nodes)]
        for col_name, col_state_list in model.states.items():
            id_dict = {state_name : state_idx for state_idx, state_name in enumerate(col_state_list)}
            samples[col_name] = list(map(lambda x : id_dict[x], samples[col_name]))

    return samples


def perturb_data(samples, card_arr, noise_model, p_noise, n_bn = 20):

    assert(noise_model in ['huber', 'independent', 'dependent', 'noisefree'])

    num_rows, num_cols = samples.shape
    column_names = list(samples.columns)
    ori_cards = card_arr
    print(ori_cards)
    if noise_model == 'huber':
        mask = np.random.rand(num_rows) < p_noise
        rand_bn_idx = np.array([np.random.choice(np.arange(n_bn), p = random_simplex(n_bn)) for _ in range(num_rows)], dtype = int)
        rand_mat = np.zeros(samples.shape)
        for bn_i in range(n_bn):
            card_list = np.random.randint(2, 10, size = num_cols)
            rand_bn = BayesianNetwork.get_random(n_nodes = num_cols, edge_prob = np.random.rand() * (2 / num_cols), n_states = card_list)
            sub_mask = (mask * (rand_bn_idx == bn_i))
            n_sub_samples = sub_mask.sum()
            rand_mat[sub_mask] = sampling(rand_bn, n_sub_samples)
        samples.loc[mask] = rand_mat[mask]
    elif noise_model == 'independent':
        mask = np.random.rand(num_rows, num_cols) < p_noise
        rand_bn_idx = np.array([np.random.choice(np.arange(n_bn), p = random_simplex(n_bn)) for _ in range(mask.size)], dtype = int).reshape(num_rows, num_cols)
        for bn_i in range(n_bn):
            card_list = np.random.randint(2, 10, size = num_cols)
            rand_bn = BayesianNetwork.get_random(n_nodes = num_cols, edge_prob = np.random.rand() * (2 / num_cols), n_states = card_list)
            sub_mask = (mask * (rand_bn_idx == bn_i))
            n_sub_samples = sub_mask.sum()
            sub_samples = sampling(rand_bn, n_sub_samples)
            for idx, coord_tup in enumerate(zip(*list(np.where(sub_mask)))):
                x, y = coord_tup
                cur_val = samples.at[x, column_names[y]]
                new_val = np.random.choice([sub_samples.at[idx, y], np.random.randint(ori_cards[column_names[y]])])
                if new_val == cur_val:
                    new_val = np.random.choice([*np.arange(cur_val), *np.arange(cur_val + 1, ori_cards[column_names[y]])])
                samples.at[x, column_names[y]] = new_val
    elif noise_model == 'dependent':
        rand_bn = BayesianNetwork.get_random(n_nodes = num_cols, edge_prob = 0.2, n_states = list(card_dict.values()))
        rand_mat = sampling(rand_bn, num_rows)
        mask = np.random.rand(num_rows, num_cols) < p_noise
        samples_np = samples.to_numpy()
        rand_mat_np = rand_mat.to_numpy()
        samples_np[mask] = rand_mat_np[mask]
        samples[:] = samples_np
    else:
        pass

    return samples


def process_df_data(data):

    data_np = data.to_numpy()
    n_rows, n_cols = data_np.shape
    cards = np.zeros(n_cols, dtype = int)

    for col_i in range(n_cols):
        cur_col = data_np[:, col_i]
        uni_ids = np.unique(cur_col)
        cards[col_i] = uni_ids.size
        id_dict = {v : k for k, v in enumerate(uni_ids)}
        data_np[:, col_i] = list(map(lambda x : id_dict[x], cur_col))
    data_np = data_np.astype(int)
    cards = cards.clip(min = 2)

    return data_np, cards


def get_csv_bn_data(data_path):

    csv_file = csv.reader(open(data_path))
    rows = [row for row in csv_file]
    data_np = np.array(rows, dtype = str)
    data_df = pd.DataFrame(data_np)

    data_np, cards = process_df_data(data_df)
    data_df = pd.DataFrame(data_np)

    return data_df, cards


def get_csv_bn_data_new(dataset_name, data_folder):

    splits = load_train_valid_test_csvs(dataset_name, data_folder)
    comb = np.concatenate(splits)
    
    data_np = np.array(comb, dtype = str)
    data_df = pd.DataFrame(data_np)

    data_np, cards = process_df_data(data_df)
    data_df = pd.DataFrame(data_np)

    return data_df, cards


def split_dataset(data_df, train_ratio = 0.5):

    n_data = data_df.shape[0]
    n_train = int(n_data * train_ratio)
    n_test = n_data - n_train

    np.random.shuffle(data_df.values)
    
    train_data = data_df.iloc[:n_train]
    test_data = data_df.iloc[n_train:]

    return train_data, test_data


def csv_2_numpy(filename, path, sep=',', type='int8'):
    """
    Utility to read a dataset in csv format into a numpy array
    """
    file_path = os.path.join(path, filename)
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    array = np.array(x).astype(type)
    return array


def load_train_valid_test_csvs(dataset_name,
                               path,
                               sep=',',
                               type='int32',
                               suffix='data',
                               splits=['train',
                                       'valid',
                                       'test'],
                               verbose=True):
    """
    Loading training, validation and test splits by suffix from csv files
    """

    csv_files = ['{0}.{1}.{2}'.format(dataset_name, ext, suffix) for ext in splits]

    dataset_splits = [csv_2_numpy(file, path, sep, type) for file in csv_files]

    return dataset_splits

