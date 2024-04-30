import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from scipy.stats import pearsonr
import collections


''' Declare constants '''
SAMPLE_SIZES = [48]  # one layer graphsage
all_nodes = np.arange(456).astype(np.int32)  # if using permutation, should change the order of y true in evaluation

TOTAL_NODES = 456
EPOCH_TRAIN = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 24
N_STEPS = 183
TESTING_SAMPLES = int(14823 / 3)
VALIDATION_SAMPLES = int(14823 / 3 * 2 / 3)
print(TESTING_SAMPLES)
print(VALIDATION_SAMPLES)

''' Load data '''
feat = np.load('src_data/processed_features.npy')
obs = np.load('src_data/obs_temp.npy').astype('float32')

adj_up = np.load('src_data/up_full.npy')
adj_dn = np.load('src_data/dn_full.npy')

print(feat.shape)
print(obs.shape)
print(adj_up.shape)
print(adj_dn.shape)

adj = adj_up + adj_dn
mean_adj = np.mean(adj[adj != 0])
std_adj = np.std(adj[adj != 0])
adj[adj != 0] = adj[adj != 0] - mean_adj
adj[adj != 0] = adj[adj != 0] / std_adj
adj[adj != 0] = 1 / (1 + np.exp(adj[adj != 0]))

A_hat = adj.copy()
A_hat[A_hat == np.nan] = 0
D = np.sum(A_hat, axis=1)
D[D == 0] = 1
D_inv = D ** -1.0
D_inv = np.diag(D_inv)
A_hat = np.matmul(D_inv, A_hat).astype('float32')
print(A_hat.shape)
print(np.sum(A_hat, axis=1))
print(np.where(np.sum(A_hat, axis=1)==0))

neigh_dict = np.int32(np.array(A_hat > 0))


# Models

initializer = tf.keras.initializers.RandomNormal(stddev=0.1)

class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, init_graph, src_dim, dst_dim, activ=True, **kwargs):
        """
        src_dim: input dimension
        dst_dim: output dimentsion
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
        self.w = self.add_weight(name=kwargs["name"] + "_weight", shape=(src_dim * 2, dst_dim), dtype=tf.float32, initializer=initializer, trainable=True)
        self.graph_w = self.add_weight(name='graph_weight', shape=(TOTAL_NODES,TOTAL_NODES), dtype=tf.float32, initializer=tf.constant_initializer(init_graph), trainable=False)

    def call(self, dstsrc_features, src_node, dstsrc2src, dstsrc2dst, dif_mat):
        """
        dstsrc_features: previous aggregation embeddings
        dstsrc2dst: current nodes for aggreagation
        dstsrc2src: all neighbors for current nodes
        dif_mat: weights matrix
        """
        # target nodes features
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        # neighbors features
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        # neighbor aggregation

        graph_weight = tf.gather(self.graph_w, tf.squeeze(src_node))
        graph_weight = tf.gather(graph_weight, dstsrc2dst)
        graph_weight = tf.transpose(graph_weight)  # Transpose to get columns
        graph_weight = tf.gather(graph_weight, dstsrc2src)  # Get selected columns
        graph_weight = tf.transpose(graph_weight)  # Transpose back to original orientation
        dif_mat = tf.math.multiply(graph_weight, dif_mat)

        num_neighbours = K.sum(tf.squeeze(dif_mat), axis=1)
        sum_aggregation = tf.matmul(dif_mat, tf.reshape(src_features, (len(dstsrc2src), N_STEPS * 20)))
        aggregated_features = tf.math.divide_no_nan(sum_aggregation, num_neighbours[:, None])

        # concatenate aggregations
        concatenated_features = tf.concat([dst_features, tf.reshape(aggregated_features, (len(dstsrc2dst), N_STEPS, 20))], 2)
        # transformation w
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)

class GraphSageBase(tf.keras.Model):

    def __init__(self, init_graph, internal_dim, num_layers, last_has_activ):  # raw_features

        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.seq_layers = []
        for i in range (1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = 20
            has_activ = last_has_activ if i == num_layers else True
            aggregator_layer = MeanAggregator(init_graph, input_dim, internal_dim, name=layer_name, activ = has_activ)
            self.seq_layers.append(aggregator_layer)

    def call(self, input_x, minibatch):
        x = tf.gather(input_x, tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            x = aggregator_layer(x, minibatch.src_nodes, minibatch.dstsrc2srcs.pop(), minibatch.dstsrc2dsts.pop(), minibatch.dif_mats.pop())
        return x

class GraphSageGlobal(GraphSageBase):
    def __init__(self, init_graph, internal_dim, num_layers): # raw_features
        super().__init__(init_graph, internal_dim, num_layers, True) # raw_features

        self.lstm_layer = tf.keras.layers.GRU(20, activation='tanh', kernel_initializer=initializer, return_sequences=True)
        self.output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer)

    def call(self, input_x, minibatch):
        layer_output = self.lstm_layer(input_x)
        out = self.output_layer( super().call(layer_output, minibatch) )
        return out


''' Split dataset '''
feat_train_all = feat[:, :-TESTING_SAMPLES]
obs_train_all = obs[:, :-TESTING_SAMPLES]
feat_test = feat[:, -TESTING_SAMPLES:]
obs_test = obs[:, -TESTING_SAMPLES:]
feat_val = feat_train_all[:, -VALIDATION_SAMPLES:]
obs_val = obs_train_all[:, -VALIDATION_SAMPLES:]
feat_tr = feat_train_all[:, :-VALIDATION_SAMPLES]
obs_tr = obs_train_all[:, :-VALIDATION_SAMPLES]

print(feat_train_all.shape, feat_tr.shape, feat_val.shape, feat_test.shape)
print(obs_train_all.shape, obs_tr.shape, obs_val.shape, obs_test.shape)
print(np.sum(obs_tr==-11), np.sum(obs_val==-11), np.sum(obs_test==-11))

''' Evaluation Functions '''
def root_mean_squared_error(y_true_in, y_pred_in):
    y_true_in = tf.reshape(y_true_in, [-1, 1])
    y_pred_in = tf.reshape(y_pred_in, [-1, 1])
    nan_mask = (y_true_in == -11)
    return K.sqrt(K.mean(K.square(y_pred_in[~nan_mask] - y_true_in[~nan_mask])))

def get_y_pred(model_in, x_in):
    y_pred_in = None
    for each_batch_in in range(int(x_in.shape[1] / N_STEPS)):
        c_y_pred = model_in(x_in[:, each_batch_in * N_STEPS:(each_batch_in + 1) * N_STEPS, :], build_batch_from_nodes(all_nodes, neigh_dict, SAMPLE_SIZES))
        if y_pred_in is None:
            y_pred_in = c_y_pred.numpy().copy()
        else:
            y_pred_in = np.concatenate([y_pred_in, c_y_pred.numpy()], 1)
    return y_pred_in

def rmse_evaluation(model_in, x_in, y_in):
    y_pred_values = get_y_pred(model_in, x_in)
    y_true_in = np.squeeze(y_in)
    y_pred_in = np.squeeze(y_pred_values)
    rmse_list_in = np.array([np.nan] * TOTAL_NODES)
    for item in range(TOTAL_NODES):
        item_true = y_true_in[item]
        item_pred = y_pred_in[item]
        nan_mask = (item_true == -11)
        if np.sum(~nan_mask) < 10:  # Ignore the small sample size
            continue
        rmse_list_in[item] = np.sqrt(np.mean(np.square(item_true[~nan_mask] - item_pred[~nan_mask])))

    y_true_in = np.reshape(y_true_in, [-1, 1])
    y_pred_in = np.reshape(y_pred_in, [-1, 1])
    nan_mask = (y_true_in == -11)
    overall_rmse_in = np.sqrt(np.mean(np.square(y_pred_in[~nan_mask] - y_true_in[~nan_mask])))

    return overall_rmse_in, rmse_list_in, np.squeeze(y_pred_values)


''' Training Functions '''

# diffusion matrix for selected nodes
def _compute_diffusion_matrix(dst_nodes, neigh_dict_in, sample_size, max_node_id):
    # random select SAMPLE_SIZE neighbors for each node
    def sample(ns):
        indices = np.where(ns==1)[0]
        sampled_indices = np.random.choice(indices, min(len(indices), sample_size), replace=False)
        return sampled_indices
    # adj vectors
    def vectorize(ns):
        v = np.zeros(max_node_id, dtype=np.float32)
        v[ns] = 1
        return v

    # adj matrix
    adj_mat_full = np.stack([vectorize(sample(neigh_dict_in[n])) for n in dst_nodes])
    # non-zero columns
    nonzero_cols_mask = np.any(adj_mat_full.astype(bool), axis=0)
    # filter adj matrx columns
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    # normalize
    dif_mat = adj_mat

    # all neighbors index for target nodes
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]

    # all nodes indices
    dstsrc = np.union1d(dst_nodes, src_nodes)

    # all neighbors indices in dstsrc
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)

    # all targets indices in dstsrc
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


# create mini batch for nodes
def build_batch_from_nodes(nodes, neigh_dict_in, sample_sizes):
    """
    nodes: nodes for current batch
    neigh_dict:
    sample_sizes: sample size for each aggregation layer
    """

    dst_nodes = [nodes]
    dstsrc2dsts = []
    dstsrc2srcs = []
    dif_mats = []

    max_node_id = TOTAL_NODES
    for sample_size in reversed(sample_sizes):
        ds, d2s, d2d, dm = _compute_diffusion_matrix(dst_nodes[-1], neigh_dict_in, sample_size, max_node_id
                                                     )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    src_nodes = dst_nodes.pop()

    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)


def generate_training_minibatch(nodes_for_training):
    for i in range(20):
        mini_batch_nodes = np.random.choice(nodes_for_training, size=BATCH_SIZE, replace=False)
        batch = build_batch_from_nodes(mini_batch_nodes, neigh_dict, SAMPLE_SIZES)
        yield (batch, mini_batch_nodes)


# start training
model_X = GraphSageGlobal(A_hat, 20, len(SAMPLE_SIZES))
optimizer_X = keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE)
model_X.compile(optimizer=optimizer_X, loss=root_mean_squared_error)

stop_sign=100

for init_epoch_number in range(EPOCH_TRAIN):
    for each_time_batch in range(int(feat_tr.shape[1] / N_STEPS)):
        minibatch_generator = generate_training_minibatch(all_nodes.copy())
        for input_graph, sampled_nodes in minibatch_generator:
            with tf.GradientTape() as tape:
                predicted = model_X(feat_tr[:, each_time_batch * N_STEPS:(each_time_batch + 1) * N_STEPS, :], input_graph)
                loss = root_mean_squared_error(tf.convert_to_tensor(obs_tr[sampled_nodes, each_time_batch * N_STEPS:(each_time_batch + 1) * N_STEPS]), predicted)
            grads = tape.gradient(loss, model_X.trainable_weights)
            optimizer_X.apply_gradients(zip(grads, model_X.trainable_weights))

    rmse_tr, _, _ = rmse_evaluation(model_X, feat_tr, obs_tr)
    rmse_test, rmse_list_test, _ = rmse_evaluation(model_X, feat_test, obs_test)
    print(
        "epoch: {:2d}  evaluation - tr_rmse: {:.6f} - test_rmse: {:.6f}".format(
            init_epoch_number, rmse_tr, rmse_test))

RMSE_BASE_TR, RMSE_LIST_TR, _ = rmse_evaluation(model_X, feat_tr, obs_tr)
RMSE_BASE_TEST, RMSE_LIST_TEST, PRED_TEST = rmse_evaluation(model_X, feat_test, obs_test)
RMSE_BASE_VAL, RMSE_LIST_VAL, _ = rmse_evaluation(model_X, feat_val, obs_val)

print('TRAINING RMSE ', RMSE_BASE_TR)
print('VALIDATING RMSE ', RMSE_BASE_VAL)
print('TESTING RMSE ', RMSE_BASE_TEST)

nan_mask1 = np.isnan(RMSE_LIST_TR)
nan_mask2 = np.isnan(RMSE_LIST_TEST[~nan_mask1])
correlation, _ = pearsonr(RMSE_LIST_TR[~nan_mask1][~nan_mask2], RMSE_LIST_TEST[~nan_mask1][~nan_mask2])
print('Correlation TR TEST ', correlation)

nan_mask1 = np.isnan(RMSE_LIST_VAL)
nan_mask2 = np.isnan(RMSE_LIST_TEST[~nan_mask1])
correlation, _ = pearsonr(RMSE_LIST_VAL[~nan_mask1][~nan_mask2], RMSE_LIST_TEST[~nan_mask1][~nan_mask2])
print('Correlation VAL TEST ', correlation)

print(np.sum(model_X.weights[1], axis=1))
print(np.where(np.sum(np.clip(model_X.weights[1], 0, 1), axis=1)==0))
