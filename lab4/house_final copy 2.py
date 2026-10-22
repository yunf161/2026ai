import numpy as np
import matplotlib.pyplot as plt
import logging
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
)
logger = logging.getLogger(__name__)
MAX_PRICE = 500001.0

def readcsv(filename):
    with open (filename, 'r') as f:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)  
    return data

def loss_function(y_pred,y):
    return np.mean((y_pred-y)**2)/2

def build_price_tier_weights(y_raw):
    y_flat = y_raw.reshape(-1)
    p80 = np.percentile(y_flat, 80)
    p90 = np.percentile(y_flat, 90)
    w = np.ones_like(y_raw, dtype=float)
    w[(y_raw >= p80) & (y_raw < p90)] = 2.0
    w[y_raw >= p90] = 3.0
    return w, p80, p90

def sigmoid(x_matmul):
    return 1/(1+np.exp(-x_matmul))

def relu(x_matmul):
    return np.maximum(0,x_matmul)

def liner(x_matmul):
    return x_matmul

def standardize(x):
    eps = 1e-8
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    x_std = (x - mu) / (std + eps) 
    return x_std, mu, std

def reverse_standardize(x_std, mu, std):
    return x_std * std + mu

def get_price_cap_std(mu_Y, std_Y, max_price=MAX_PRICE):
    return (np.log1p(max_price) - mu_Y) / (std_Y + 1e-8)

def report_error_metrics(y_true, y_pred, stage_name):
    # Use original unit metrics so we can interpret how many currency units the model misses.
    eps = 1e-8
    residual = y_true - y_pred
    abs_err = np.abs(residual)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(abs_err)
    p90_abs = np.percentile(abs_err, 90)
    avg_acc = np.mean(1 - np.abs(residual) / (np.abs(y_true) + eps))
    y_true_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2) + eps
    r2 = 1 - ss_res / ss_tot
    logger.info("%s RMSE: %.4f", stage_name, rmse)
    logger.info("%s MAE : %.4f", stage_name, mae)
    logger.info("%s P90 : %.4f", stage_name, p90_abs)
    logger.info("%s AvgAcc(1-|residual/actual|): %.4f", stage_name, avg_acc)
    logger.info("%s R2  : %.4f", stage_name, r2)
    return residual, rmse, mae, p90_abs, avg_acc, r2

def fit_kmeans_lonlat(lonlat_train, n_clusters=8, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = lonlat_train.shape[0]
    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = lonlat_train[init_idx].copy()

    for _ in range(max_iter):
        dists = np.sum((lonlat_train[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for k in range(n_clusters):
            cluster_points = lonlat_train[labels == k]
            if cluster_points.shape[0] > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return centroids

def assign_kmeans_cluster(lonlat, centroids):
    dists = np.sum((lonlat[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1)

def preprocess_income_tail(data_train, data_val, data_test, income_idx=3, low_q=0.01, high_q=0.99):
    """Winsorize income by train quantiles to stabilize tail behavior."""
    low = np.percentile(data_train[:, income_idx], low_q * 100)
    high = np.percentile(data_train[:, income_idx], high_q * 100)

    train_new = data_train.copy()
    val_new = data_val.copy()
    test_new = data_test.copy()
    train_new[:, income_idx] = np.clip(train_new[:, income_idx], low, high)
    val_new[:, income_idx] = np.clip(val_new[:, income_idx], low, high)
    test_new[:, income_idx] = np.clip(test_new[:, income_idx], low, high)

    logger.info(
        "Income winsorization enabled on feature_%d: low_q=%.2f(%.4f), high_q=%.2f(%.4f)",
        income_idx,
        low_q,
        low,
        high_q,
        high,
    )
    return train_new, val_new, test_new, low, high

def append_geo_cluster_feature(data_split, centroids, capped_income_threshold, lonlat_scale=0.2, cluster_scale=2.0):
    x = data_split[:, :-1]
    y = data_split[:, -1:]
    lonlat = x[:, :2]
    cluster_id = assign_kmeans_cluster(lonlat, centroids)
    cluster_center = centroids[cluster_id]
    dx = lonlat[:, 0:1] - cluster_center[:, 0:1]
    dy = lonlat[:, 1:2] - cluster_center[:, 1:2]
    r = np.sqrt(dx ** 2 + dy ** 2)
    local_geo = np.concatenate([dx, dy, r], axis=1)

    # Attenuate raw lon/lat so the model relies less on exact coordinates.
    x_weighted = x.copy()
    x_weighted[:, :2] = x_weighted[:, :2] * lonlat_scale

    # Use an input-only proxy flag for capped-price risk to avoid target leakage.
    capped_proxy_flag = (x[:, 3:4] >= capped_income_threshold).astype(float)

    # Use one-hot cluster features and amplify them to prioritize region signal.
    n_clusters = centroids.shape[0]
    cluster_one_hot = np.eye(n_clusters)[cluster_id].astype(float) * cluster_scale

    x_aug = np.concatenate([x_weighted, local_geo, capped_proxy_flag, cluster_one_hot], axis=1)
    return np.concatenate([x_aug, y], axis=1)

def append_multi_cluster_features(data_split, geo_centroids, extra_centroids_by_feature, lonlat_scale=0.2, cluster_scale=2.0):
    x = data_split[:, :-1]
    y = data_split[:, -1:]

    # Keep raw features but weaken lon/lat magnitude.
    x_weighted = x.copy()
    x_weighted[:, :2] = x_weighted[:, :2] * lonlat_scale

    # lon/lat 2D cluster one-hot
    geo_labels = assign_kmeans_cluster(x[:, :2], geo_centroids)
    geo_one_hot = np.eye(geo_centroids.shape[0])[geo_labels].astype(float) * cluster_scale

    extra_one_hots = []
    for feat_idx, centroids in extra_centroids_by_feature.items():
        feat_col = x[:, feat_idx:feat_idx + 1]
        feat_labels = assign_kmeans_cluster(feat_col, centroids)
        feat_one_hot = np.eye(centroids.shape[0])[feat_labels].astype(float) * cluster_scale
        extra_one_hots.append(feat_one_hot)

    if extra_one_hots:
        x_aug = np.concatenate([x_weighted, geo_one_hot] + extra_one_hots, axis=1)
    else:
        x_aug = np.concatenate([x_weighted, geo_one_hot], axis=1)

    return np.concatenate([x_aug, y], axis=1)

def price_bin_labels(y, step=50000):
    y_flat = y.reshape(-1)
    return np.floor(y_flat / step).astype(int)

def log_price_bin_distribution(y, stage_name, step=50000):
    labels = price_bin_labels(y, step=step)
    unique_bins, counts = np.unique(labels, return_counts=True)
    logger.info("%s price-bin distribution (step=%d):", stage_name, step)
    for b, c in zip(unique_bins, counts):
        low = b * step
        high = (b + 1) * step
        logger.info("  [%d, %d): %d", low, high, int(c))

def report_metrics_by_price_bin(y_true, y_pred, stage_name, step=50000):
    labels = price_bin_labels(y_true, step=step)
    logger.info("%s metrics by price-bin (step=%d):", stage_name, step)
    for b in np.unique(labels):
        mask = labels == b
        if np.sum(mask) < 20:
            continue
        residual = y_true[mask] - y_pred[mask]
        rmse = np.sqrt(np.mean(residual ** 2))
        mae = np.mean(np.abs(residual))
        low = b * step
        high = (b + 1) * step
        logger.info("  [%d, %d): count=%d, RMSE=%.4f, MAE=%.4f", low, high, int(np.sum(mask)), rmse, mae)

def diagnose_split_distribution(data_test, data_val=None, data_train=None, top_k_features=4):
    # New mode (recommended): diagnose_split_distribution(test, val, train)
    # Backward-compatible mode: diagnose_split_distribution(train, test)
    if data_train is None and data_val is not None:
        # Old 2-way call detected: first is train, second is test.
        data_train, data_test = data_test, data_val
        data_val = None

    if data_train is None:
        raise ValueError("Please provide (test, val, train) or (train, test).")

    x_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    x_test = data_test[:, :-1]
    y_test = data_test[:, -1]

    def _summary(arr):
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p50': float(np.percentile(arr, 50)),
            'p90': float(np.percentile(arr, 90)),
        }

    logger.info("===== Split Diagnosis =====")
    y_train_s = _summary(y_train)
    y_test_s = _summary(y_test)

    if data_val is None:
        logger.info("Target(y) train/test summary:")
        logger.info("  train mean=%.3f, std=%.3f, p10/p50/p90=%.3f/%.3f/%.3f", y_train_s['mean'], y_train_s['std'], y_train_s['p10'], y_train_s['p50'], y_train_s['p90'])
        logger.info("  test  mean=%.3f, std=%.3f, p10/p50/p90=%.3f/%.3f/%.3f", y_test_s['mean'], y_test_s['std'], y_test_s['p10'], y_test_s['p50'], y_test_s['p90'])

        mean_gap = abs(y_train_s['mean'] - y_test_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
        std_gap = abs(y_train_s['std'] - y_test_s['std']) / (abs(y_train_s['std']) + 1e-8)
        logger.info("Target relative gap: mean_gap=%.3f, std_gap=%.3f", mean_gap, std_gap)

        feat_gaps = []
        for j in range(x_train.shape[1]):
            tr_m = float(np.mean(x_train[:, j]))
            te_m = float(np.mean(x_test[:, j]))
            tr_s = float(np.std(x_train[:, j]))
            te_s = float(np.std(x_test[:, j]))
            gap = abs(tr_m - te_m) / (abs(tr_m) + 1e-8)
            feat_gaps.append((j, gap, tr_m, te_m, tr_s, te_s))
        feat_gaps.sort(key=lambda x: x[1], reverse=True)

        logger.info("Top %d feature gaps:", min(top_k_features, len(feat_gaps)))
        for j, gap, tr_m, te_m, tr_s, te_s in feat_gaps[:top_k_features]:
            logger.info("  feature_%d: mean_gap=%.3f, train_mean=%.3f, test_mean=%.3f, train_std=%.3f, test_std=%.3f", j, gap, tr_m, te_m, tr_s, te_s)
        logger.info("===== End Diagnosis =====")
        return

    x_val = data_val[:, :-1]
    y_val = data_val[:, -1]
    y_val_s = _summary(y_val)

    logger.info("Target(y) test/val/train summary:")
    logger.info("  test  mean=%.3f, std=%.3f, p10/p50/p90=%.3f/%.3f/%.3f", y_test_s['mean'], y_test_s['std'], y_test_s['p10'], y_test_s['p50'], y_test_s['p90'])
    logger.info("  val   mean=%.3f, std=%.3f, p10/p50/p90=%.3f/%.3f/%.3f", y_val_s['mean'], y_val_s['std'], y_val_s['p10'], y_val_s['p50'], y_val_s['p90'])
    logger.info("  train mean=%.3f, std=%.3f, p10/p50/p90=%.3f/%.3f/%.3f", y_train_s['mean'], y_train_s['std'], y_train_s['p10'], y_train_s['p50'], y_train_s['p90'])

    mean_gap_test = abs(y_train_s['mean'] - y_test_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
    std_gap_test = abs(y_train_s['std'] - y_test_s['std']) / (abs(y_train_s['std']) + 1e-8)
    mean_gap_val = abs(y_train_s['mean'] - y_val_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
    std_gap_val = abs(y_train_s['std'] - y_val_s['std']) / (abs(y_train_s['std']) + 1e-8)
    logger.info("Target relative gap (test vs train): mean_gap=%.3f, std_gap=%.3f", mean_gap_test, std_gap_test)
    logger.info("Target relative gap (val  vs train): mean_gap=%.3f, std_gap=%.3f", mean_gap_val, std_gap_val)

    feat_gaps_test = []
    feat_gaps_val = []
    for j in range(x_train.shape[1]):
        tr_m = float(np.mean(x_train[:, j]))
        te_m = float(np.mean(x_test[:, j]))
        va_m = float(np.mean(x_val[:, j]))
        tr_s = float(np.std(x_train[:, j]))
        te_s = float(np.std(x_test[:, j]))
        va_s = float(np.std(x_val[:, j]))
        gap_test = abs(tr_m - te_m) / (abs(tr_m) + 1e-8)
        gap_val = abs(tr_m - va_m) / (abs(tr_m) + 1e-8)
        feat_gaps_test.append((j, gap_test, tr_m, te_m, tr_s, te_s))
        feat_gaps_val.append((j, gap_val, tr_m, va_m, tr_s, va_s))

    feat_gaps_test.sort(key=lambda x: x[1], reverse=True)
    feat_gaps_val.sort(key=lambda x: x[1], reverse=True)

    logger.info("Top %d feature gaps (test vs train):", min(top_k_features, len(feat_gaps_test)))
    for j, gap, tr_m, te_m, tr_s, te_s in feat_gaps_test[:top_k_features]:
        logger.info("  feature_%d: mean_gap=%.3f, test_mean=%.3f, train_mean=%.3f, test_std=%.3f, train_std=%.3f", j, gap, te_m, tr_m, te_s, tr_s)

    logger.info("Top %d feature gaps (val vs train):", min(top_k_features, len(feat_gaps_val)))
    for j, gap, tr_m, va_m, tr_s, va_s in feat_gaps_val[:top_k_features]:
        logger.info("  feature_%d: mean_gap=%.3f, val_mean=%.3f, train_mean=%.3f, val_std=%.3f, train_std=%.3f", j, gap, va_m, tr_m, va_s, tr_s)
    logger.info("===== End Diagnosis =====")

def house_price_prediction_3layers_predict(data,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y):
    data = data.astype(float)

    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    Y=np.log1p(Y)
    
    # 标准化输入数据
    X = (X - mu_X) / (std_X + 1e-8)
    # 标准化目标变量
    Y = (Y - mu_Y) / (std_Y + 1e-8)
    
     #隐藏层节点
    hidden_layers=np.matmul(X,weights)+offsets #n*nodes
    hidden_layers = relu(hidden_layers) #n*nodes
    hidden_layers2=np.matmul(hidden_layers,weights2)+offsets2 #n*nodess2
    hidden_layers2 = relu(hidden_layers2) #n*nodess2
    y_cap_std = get_price_cap_std(mu_Y, std_Y)
    #输出节点
    z_out = hidden_layers2 @ weights3 + offset3   # (n, 1)
    Y_pred_raw = liner(z_out) #n*1
    Y_pred = np.minimum(Y_pred_raw, y_cap_std)
    loss=loss_function(Y_pred,Y)
   

        #输出节点
  
    return Y_pred,loss

def house_price_prediction_3layers_train(data,data_foucus,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=300000,Max_patience=10000000,patience=0.0000000001,nodes=64):
    

    data = data.astype(float)
    data_foucus = data_foucus.astype(float)
    
    X=data[:,:-1] #n*features
    Y=data[:,-1:]  #n*1
    sample_weights, p80, p90 = build_price_tier_weights(Y)
    logger.info("Tiered sample weights enabled: y<p80 => 1, p80<=y<p90 => 2, y>=p90 => 3")
    logger.info("Weight thresholds: p80=%.2f, p90=%.2f", p80, p90)
    Y=np.log1p(Y)
    
    X, mu_X, std_X = standardize(X)
    # 标准化目标变量
    Y, mu_Y, std_Y = standardize(Y)
    y_cap_std = get_price_cap_std(mu_Y, std_Y)

    n_features = X.shape[1]
    weights = np.random.randn(n_features, nodes) * np.sqrt(2.0 / n_features)
    offsets=np.zeros(nodes) #nodes

    nodess2=nodes//2
    weights2 = np.random.randn(nodes, nodess2) * np.sqrt(2.0 / nodes) #nodes*nodess2
    offsets2=np.zeros(nodess2) #nodess2
   
    #输出节点
    weights3 = np.random.randn(nodess2, 1) * np.sqrt(2.0 / nodess2) #nodess2*1
    offset3 = 0.0

    # Save best checkpoint on validation improvements for true early-stopping rollback.
    best_weights = weights.copy()
    best_offsets = offsets.copy()
    best_weights2 = weights2.copy()
    best_offsets2 = offsets2.copy()
    best_weights3 = weights3.copy()
    best_offset3 = float(offset3)

    Y_pred = np.zeros(Y.shape)
    loss=100
    count=0
    best_focus_loss = np.inf
    best_iter = 0
    patience_count=0
    iter_hist = []
    train_loss_hist = []
    val_loss_hist = []

    
    while loss>0.01 and count<Max_iterations:
        # 验证集损失早停检查
        if(count%1000==0 and count!=0):
            logger.info("Iteration %d,patience_count=%d", count, patience_count)
            Y_train_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
            Y_train_pred_original=np.expm1(Y_train_pred_original)
            y_train_true = data[:,-1:]
            loss_list_y, RMSE, MAE, P90, AVG_ACC, R2 = report_error_metrics(y_train_true, Y_train_pred_original, "Training")
            logger.info("training loss=%.4f", loss)
            #print(f"Iteration {count}, loss={loss-0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights3**2)):.4f}")
            Y_foucus_pred ,Y_foucus_loss= house_price_prediction_3layers_predict(data_foucus,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
            Y_foucus_loss += 0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights2**2) + np.sum(weights3**2))
            Y_foucus_pred_original = reverse_standardize(Y_foucus_pred, mu_Y, std_Y)
            Y_foucus_pred_original=np.expm1(Y_foucus_pred_original)
            y_val_true = data_foucus[:,-1:]
            _, val_rmse, val_mae, val_p90, val_avg_acc, val_r2 = report_error_metrics(y_val_true, Y_foucus_pred_original, "Validation")
            logger.info("validation loss=%.4f", Y_foucus_loss)
            iter_hist.append(count)
            train_loss_hist.append(loss)
            val_loss_hist.append(Y_foucus_loss)
            if Y_foucus_loss < best_focus_loss - patience:
                patience_count=0
                best_focus_loss = Y_foucus_loss
                best_iter = count
                best_weights = weights.copy()
                best_offsets = offsets.copy()
                best_weights2 = weights2.copy()
                best_offsets2 = offsets2.copy()
                best_weights3 = weights3.copy()
                best_offset3 = float(offset3)
            else:
                patience_count+=1
                if patience_count>Max_patience:
                    weights = best_weights.copy()
                    offsets = best_offsets.copy()
                    weights2 = best_weights2.copy()
                    offsets2 = best_offsets2.copy()
                    weights3 = best_weights3.copy()
                    offset3 = best_offset3

                    Y_foucus_pred_best, _ = house_price_prediction_3layers_predict(data_foucus,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
                    Y_foucus_pred_original = reverse_standardize(Y_foucus_pred_best, mu_Y, std_Y)
                    Y_foucus_pred_original=np.expm1(Y_foucus_pred_original)
                    y_train_true = data_foucus[:,-1:]
                    loss_list_y, RMSE, MAE, P90, AVG_ACC, R2 = report_error_metrics(y_train_true, Y_foucus_pred_original, "Validation")
                    logger.info("Early stopping at iteration %d, best_iter=%d, best_focus_loss=%.4f", count, best_iter, best_focus_loss)

                    hidden_layers_best=np.matmul(X,weights)+offsets
                    hidden_layers_best = relu(hidden_layers_best)
                    hidden_layers2_best=np.matmul(hidden_layers_best,weights2)+offsets2
                    hidden_layers2_best = relu(hidden_layers2_best)
                    Y_pred_best_raw = liner(hidden_layers2_best @ weights3 + offset3)
                    Y_pred_best = np.minimum(Y_pred_best_raw, y_cap_std)
                    loss_best=loss_function(Y_pred_best,Y)+0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights2**2) + np.sum(weights3**2))
                    return weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,Y_pred_best,mu_Y,std_Y,loss_best,iter_hist,train_loss_hist,val_loss_hist
            if(count%100000==0 and count!=0):
                learning_rate*=0.9
     # 前向传播
     #隐藏层节点
        hidden_layers=np.matmul(X,weights)+offsets #n*nodes
        hidden_layers = relu(hidden_layers) #n*nodes
        hidden_layers2=np.matmul(hidden_layers,weights2)+offsets2 #n*nodess2
        hidden_layers2 = relu(hidden_layers2) #n*nodess2
        #输出节点
        z_out = hidden_layers2 @ weights3 + offset3   # (n, 1)
        Y_pred_raw = liner(z_out) #n*1
        Y_pred = np.minimum(Y_pred_raw, y_cap_std)
        clip_mask = (Y_pred_raw <= y_cap_std).astype(float)
        weighted_data_loss = np.sum(sample_weights * (Y_pred - Y) ** 2) / (2 * Y.shape[0])
        loss=weighted_data_loss+0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights2**2) + np.sum(weights3**2))
       # print(loss)
        # 反向传播
        #计算各个偏导数

        dloss_func_dY_pred=(sample_weights * (Y_pred-Y))/Y.shape[0] #n*1
        dloss_func_dz_out = dloss_func_dY_pred * clip_mask
        dY_pred_dweights3=hidden_layers2 #n*nodess2
        #dY_pred_dweights=weights3*(hidden_layers>0).astype(float)*X #nodes*1 n*nodes   n*1   目标4*nodes
        #dY_pred_doffsets=weights3*(hidden_layers>0).astype(float)

        #梯度
        T_weights3=dY_pred_dweights3.T @ dloss_func_dz_out #n*nodes.T x n*1 按列平均得到1*32
        T_offsets3=np.sum(dloss_func_dz_out) #1
        dloss_dhidden2 = dloss_func_dz_out @ weights3.T * (hidden_layers2>0).astype(float)
        T_weights2=hidden_layers.T @ dloss_dhidden2 #nodes*nodess2
        T_offsets2=np.sum(dloss_dhidden2, axis=0) #nodess2
        dloss_dhidden1 = dloss_dhidden2 @ weights2.T * (hidden_layers>0).astype(float)
        T_weights=X.T @ dloss_dhidden1 #4*nodes 
        T_offsets=np.sum(dloss_dhidden1, axis=0) #nodes
        #更新权重和偏置
        weights3 -= learning_rate * (T_weights3 + l2_lambda*weights3) #nodes*1
        weights2 -= learning_rate * (T_weights2 + l2_lambda*weights2) #nodes*nodess2
        offsets2 -= learning_rate * T_offsets2
        weights -= learning_rate * (T_weights + l2_lambda*weights) #4*nodes
        offsets -= learning_rate * T_offsets
        offset3 -= learning_rate * T_offsets3
      
        count+=1
        
        
    logger.info("Training ended at iteration %d", count)
    hidden_layers_best=np.matmul(X,best_weights)+best_offsets
    hidden_layers_best = relu(hidden_layers_best)
    hidden_layers2_best=np.matmul(hidden_layers_best,best_weights2)+best_offsets2
    hidden_layers2_best = relu(hidden_layers2_best)
    Y_pred_best_raw = liner(hidden_layers2_best @ best_weights3 + best_offset3)
    Y_pred_best = np.minimum(Y_pred_best_raw, y_cap_std)
    loss_best=loss_function(Y_pred_best,Y)+0.5 * l2_lambda * (np.sum(best_weights**2) + np.sum(best_weights2**2) + np.sum(best_weights3**2))
    return best_weights,best_offsets,best_weights2,best_offsets2,best_weights3,best_offset3,mu_X,std_X,Y_pred_best,mu_Y,std_Y,loss_best,iter_hist,train_loss_hist,val_loss_hist

def carry_out(data,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=500000000,Max_patience=100000,patience=0.000000001,nodes=64,k_clusters=8,price_bin_step=50000,lonlat_scale=0.2,cluster_scale=2.0):
    np.random.seed(42)
    np.random.shuffle(data)  # data 的行顺序被打乱

    data_train=data[:6000,:] #前6000行作为训练集
    data_foucus=data[6000:8000,:] #2000行作为验证集
    data_test=data[8000:,:] #后2000行作为测试集

    # Tail preprocessing for income (train-quantile winsorization).
    data_train, data_foucus, data_test, _, income_q99 = preprocess_income_tail(
        data_train,
        data_foucus,
        data_test,
        income_idx=3,
        low_q=0.01,
        high_q=0.99,
    )

    # Keep raw lon/lat for geographic visualization (before any scaling/feature augmentation).
    test_lonlat_raw = data_test[:, :2].copy()

    # Fit lon/lat KMeans on train only to avoid leakage, then append geo-cluster one-hot feature.
    centroids = fit_kmeans_lonlat(data_train[:, :2], n_clusters=k_clusters, max_iter=100, seed=42)
    data_train = append_geo_cluster_feature(data_train, centroids, income_q99, lonlat_scale=lonlat_scale, cluster_scale=cluster_scale)
    data_foucus = append_geo_cluster_feature(data_foucus, centroids, income_q99, lonlat_scale=lonlat_scale, cluster_scale=cluster_scale)
    data_test = append_geo_cluster_feature(data_test, centroids, income_q99, lonlat_scale=lonlat_scale, cluster_scale=cluster_scale)
    logger.info("Added geo features: cluster one-hot + local dx/dy/r + capped_proxy_flag(income>=q99), k=%d", k_clusters)

    log_price_bin_distribution(data_train[:, -1:], "Train", step=price_bin_step)
    log_price_bin_distribution(data_foucus[:, -1:], "Validation", step=price_bin_step)
    log_price_bin_distribution(data_test[:, -1:], "Test", step=price_bin_step)
    

    # Diagnose whether sequential split introduces train/test distribution shift.
    diagnose_split_distribution( data_test,data_foucus,data_train, top_k_features=4)

    weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,Y_train_pred,mu_Y,std_Y,loss,iter_hist,train_loss_hist,val_loss_hist=house_price_prediction_3layers_train(data_train,data_foucus,learning_rate, l2_lambda, Max_iterations, Max_patience, patience, nodes)
    Y_train_pred_original = reverse_standardize(Y_train_pred, mu_Y, std_Y)
    Y_train_pred_original=np.expm1(Y_train_pred_original)
    y_train_true = data_train[:,-1:]
    loss_list_y, RMSE, MAE, P90, AVG_ACC, R2 = report_error_metrics(y_train_true, Y_train_pred_original, "Training")
    report_metrics_by_price_bin(y_train_true, Y_train_pred_original, "Training", step=price_bin_step)
    

    Y_pred,loss = house_price_prediction_3layers_predict(data_test,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
    Y_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
    Y_pred_original=np.expm1(Y_pred_original)
    y_test_true = data_test[:,-1:]
    loss_list_ypred, RMSE, MAE, P90, AVG_ACC, R2 = report_error_metrics(y_test_true, Y_pred_original, "Testing")
    report_metrics_by_price_bin(y_test_true, Y_pred_original, "Testing", step=price_bin_step)

    test_abs_error = np.abs(y_test_true - Y_pred_original)
    
    return (
        loss_list_y,
        loss_list_ypred,
        iter_hist,
        train_loss_hist,
        val_loss_hist,
        mu_X,
        std_X,
        mu_Y,
        std_Y,
        y_test_true,
        Y_pred_original,
        test_lonlat_raw,
        test_abs_error,
    )
  

def main():
    data=readcsv('MLP_data.csv')
  
    np.random.seed(42)
    np.random.shuffle(data) # data 的行顺序被打乱

    (
        loss_list_y,
        loss_list_ypred,
        iter_hist,
        train_loss_hist,
        val_loss_hist,
        mu_X,
        std_X,
        mu_Y,
        std_Y,
        y_test_true,
        y_test_pred,
        test_lonlat_raw,
        test_abs_error,
    ) = carry_out(data,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=50000,Max_patience=5,patience=0.00002,nodes=64)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1:3])

    ax0.scatter(np.arange(loss_list_y.shape[0]),loss_list_y.ravel(), color='blue', s=10,alpha=0.5)
    ax0.set_xlabel('Index')
    ax0.set_ylabel('Residual (Actual Price - Predicted Price)')
    ax0.set_title('Training Error Scatter Plot')
    
    ax1.scatter(np.arange(loss_list_ypred.shape[0]),loss_list_ypred.ravel(), color='red', s=10,alpha=0.5)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Residual (Actual Price - Predicted Price)')
    ax1.set_title('Testing Error Scatter Plot')

    ax2.plot(iter_hist, train_loss_hist, color='blue', label='Training Loss')
    ax2.plot(iter_hist, val_loss_hist, color='orange', label='Validation Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curve')
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Actual vs Predicted with y=x reference line for calibration/underestimation check.
    y_true_flat = y_test_true.ravel()
    y_pred_flat = y_test_pred.ravel()
    min_v = float(min(np.min(y_true_flat), np.min(y_pred_flat)))
    max_v = float(max(np.max(y_true_flat), np.max(y_pred_flat)))
    ax3.scatter(y_true_flat, y_pred_flat, s=10, alpha=0.45, color='teal')
    ax3.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', linewidth=1.2, label='y = x')
    ax3.set_xlabel('Actual Price')
    ax3.set_ylabel('Predicted Price')
    ax3.set_title('Actual vs Predicted (Test)')
    ax3.grid(alpha=0.25)
    ax3.legend()

    # Geographic error map: color by absolute error to show where prediction is weak.
    geo_sc = ax4.scatter(
        test_lonlat_raw[:, 0],
        test_lonlat_raw[:, 1],
        c=test_abs_error.ravel(),
        cmap='viridis',
        s=14,
        alpha=0.75,
    )
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_title('Geographic Error Map (Color = |Error|)')
    ax4.grid(alpha=0.2)
    cbar = fig.colorbar(geo_sc, ax=ax4, fraction=0.03, pad=0.02)
    cbar.set_label('Absolute Error')

    mu_x_txt = np.array2string(mu_X.ravel(), precision=4, separator=', ')
    std_x_txt = np.array2string(std_X.ravel(), precision=4, separator=', ')
    process_note = (
        "Data Processing:\n"
        "X: standardized by train stats\n"
        "X_std = (X - mu_X) / (std_X + 1e-8)\n"
        f"mu_X={mu_x_txt}\n"
        f"std_X={std_x_txt}\n"
        "y: log1p(y) then standardized\n"
        f"mu_Y={float(mu_Y.ravel()[0]):.4f}, std_Y={float(std_Y.ravel()[0]):.4f}\n"
        "inverse: y = expm1(y_std*std_Y + mu_Y)"
    )
    ax2.text(
        0.02,
        0.98,
        process_note,
        transform=ax2.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.tight_layout()
    plt.show()



   





if __name__ == "__main__":   
    main()