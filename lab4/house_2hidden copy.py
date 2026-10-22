import numpy as np
import matplotlib.pyplot as plt
def readcsv(filename):
    with open (filename, 'r') as f:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)  
    return data

def loss_function(y_pred,y):
    return np.mean((y_pred-y)**2)/2

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

def report_error_metrics(y_true, y_pred, stage_name):
    # Use original unit metrics so we can interpret how many currency units the model misses.
    residual = y_true - y_pred
    abs_err = np.abs(residual)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(abs_err)
    p90_abs = np.percentile(abs_err, 90)
    print(f"{stage_name} RMSE: {rmse:.4f}")
    print(f"{stage_name} MAE : {mae:.4f}")
    print(f"{stage_name} P90 : {p90_abs:.4f}")
    return residual, rmse, mae, p90_abs

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

    print("===== Split Diagnosis =====")
    y_train_s = _summary(y_train)
    y_test_s = _summary(y_test)

    if data_val is None:
        print("Target(y) train/test summary:")
        print(f"  train mean={y_train_s['mean']:.3f}, std={y_train_s['std']:.3f}, p10/p50/p90={y_train_s['p10']:.3f}/{y_train_s['p50']:.3f}/{y_train_s['p90']:.3f}")
        print(f"  test  mean={y_test_s['mean']:.3f}, std={y_test_s['std']:.3f}, p10/p50/p90={y_test_s['p10']:.3f}/{y_test_s['p50']:.3f}/{y_test_s['p90']:.3f}")

        mean_gap = abs(y_train_s['mean'] - y_test_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
        std_gap = abs(y_train_s['std'] - y_test_s['std']) / (abs(y_train_s['std']) + 1e-8)
        print(f"Target relative gap: mean_gap={mean_gap:.3f}, std_gap={std_gap:.3f}")

        feat_gaps = []
        for j in range(x_train.shape[1]):
            tr_m = float(np.mean(x_train[:, j]))
            te_m = float(np.mean(x_test[:, j]))
            tr_s = float(np.std(x_train[:, j]))
            te_s = float(np.std(x_test[:, j]))
            gap = abs(tr_m - te_m) / (abs(tr_m) + 1e-8)
            feat_gaps.append((j, gap, tr_m, te_m, tr_s, te_s))
        feat_gaps.sort(key=lambda x: x[1], reverse=True)

        print(f"Top {min(top_k_features, len(feat_gaps))} feature gaps:")
        for j, gap, tr_m, te_m, tr_s, te_s in feat_gaps[:top_k_features]:
            print(f"  feature_{j}: mean_gap={gap:.3f}, train_mean={tr_m:.3f}, test_mean={te_m:.3f}, train_std={tr_s:.3f}, test_std={te_s:.3f}")
        print("===== End Diagnosis =====")
        return

    x_val = data_val[:, :-1]
    y_val = data_val[:, -1]
    y_val_s = _summary(y_val)

    print("Target(y) test/val/train summary:")
    print(f"  test  mean={y_test_s['mean']:.3f}, std={y_test_s['std']:.3f}, p10/p50/p90={y_test_s['p10']:.3f}/{y_test_s['p50']:.3f}/{y_test_s['p90']:.3f}")
    print(f"  val   mean={y_val_s['mean']:.3f}, std={y_val_s['std']:.3f}, p10/p50/p90={y_val_s['p10']:.3f}/{y_val_s['p50']:.3f}/{y_val_s['p90']:.3f}")
    print(f"  train mean={y_train_s['mean']:.3f}, std={y_train_s['std']:.3f}, p10/p50/p90={y_train_s['p10']:.3f}/{y_train_s['p50']:.3f}/{y_train_s['p90']:.3f}")

    mean_gap_test = abs(y_train_s['mean'] - y_test_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
    std_gap_test = abs(y_train_s['std'] - y_test_s['std']) / (abs(y_train_s['std']) + 1e-8)
    mean_gap_val = abs(y_train_s['mean'] - y_val_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
    std_gap_val = abs(y_train_s['std'] - y_val_s['std']) / (abs(y_train_s['std']) + 1e-8)
    print(f"Target relative gap (test vs train): mean_gap={mean_gap_test:.3f}, std_gap={std_gap_test:.3f}")
    print(f"Target relative gap (val  vs train): mean_gap={mean_gap_val:.3f}, std_gap={std_gap_val:.3f}")

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

    print(f"Top {min(top_k_features, len(feat_gaps_test))} feature gaps (test vs train):")
    for j, gap, tr_m, te_m, tr_s, te_s in feat_gaps_test[:top_k_features]:
        print(f"  feature_{j}: mean_gap={gap:.3f}, test_mean={te_m:.3f}, train_mean={tr_m:.3f}, test_std={te_s:.3f}, train_std={tr_s:.3f}")

    print(f"Top {min(top_k_features, len(feat_gaps_val))} feature gaps (val vs train):")
    for j, gap, tr_m, va_m, tr_s, va_s in feat_gaps_val[:top_k_features]:
        print(f"  feature_{j}: mean_gap={gap:.3f}, val_mean={va_m:.3f}, train_mean={tr_m:.3f}, val_std={va_s:.3f}, train_std={tr_s:.3f}")
    print("===== End Diagnosis =====")

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
    #输出节点
    z_out = hidden_layers2 @ weights3 + offset3   # (n, 1)
    Y_pred = liner(z_out) #n*1   
    loss=loss_function(Y_pred,Y)
   

        #输出节点
  
    return Y_pred,loss

def house_price_prediction_3layers_train(data,data_foucus,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=300000,Max_patience=10000000,patience=0.0000000001,nodes=64):
    

    data = data.astype(float)
    data_foucus = data_foucus.astype(float)
    
    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    Y=np.log1p(Y)
    
    X, mu_X, std_X = standardize(X)
    # 标准化目标变量
    Y, mu_Y, std_Y = standardize(Y)
   
    weights = np.random.randn(4, nodes) * np.sqrt(2.0 / 4)#4*nodes
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
            print(f"Iteration {count},patience_count={patience_count}" )
            Y_train_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
            Y_train_pred_original=np.expm1(Y_train_pred_original)
            y_train_true = data[:,-1:]
            loss_list_y, RMSE, MAE, P90 = report_error_metrics(y_train_true, Y_train_pred_original, "Training")
            print(f"training loss={loss:.4f}")
            #print(f"Iteration {count}, loss={loss-0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights3**2)):.4f}")
            Y_foucus_pred ,Y_foucus_loss= house_price_prediction_3layers_predict(data_foucus,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
            Y_foucus_loss += 0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights2**2) + np.sum(weights3**2))
            Y_foucus_pred_original = reverse_standardize(Y_foucus_pred, mu_Y, std_Y)
            Y_foucus_pred_original=np.expm1(Y_foucus_pred_original)
            y_val_true = data_foucus[:,-1:]
            _, val_rmse, val_mae, val_p90 = report_error_metrics(y_val_true, Y_foucus_pred_original, "Validation")
            print(f"validation loss={Y_foucus_loss:.4f}")
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
                    loss_list_y, RMSE, MAE, P90 = report_error_metrics(y_train_true, Y_foucus_pred_original, "Validation")
                    print(f"Early stopping at iteration {count}, best_iter={best_iter}, best_focus_loss={best_focus_loss:.4f}")

                    hidden_layers_best=np.matmul(X,weights)+offsets
                    hidden_layers_best = relu(hidden_layers_best)
                    hidden_layers2_best=np.matmul(hidden_layers_best,weights2)+offsets2
                    hidden_layers2_best = relu(hidden_layers2_best)
                    Y_pred_best = liner(hidden_layers2_best @ weights3 + offset3)
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
        Y_pred = liner(z_out) #n*1   
        loss=loss_function(Y_pred,Y)+0.5 * l2_lambda * (np.sum(weights**2) + np.sum(weights2**2) + np.sum(weights3**2))
       # print(loss)
        # 反向传播
        #计算各个偏导数

        dloss_func_dY_pred=(Y_pred-Y)/Y.shape[0] #n*1
        dY_pred_dweights3=hidden_layers2 #n*nodess2
        #dY_pred_dweights=weights3*(hidden_layers>0).astype(float)*X #nodes*1 n*nodes   n*1   目标4*nodes
        #dY_pred_doffsets=weights3*(hidden_layers>0).astype(float)

        #梯度
        T_weights3=dY_pred_dweights3.T @ dloss_func_dY_pred #n*nodes.T x n*1 按列平均得到1*32
        T_offsets3=np.sum(dloss_func_dY_pred) #1
        dloss_dhidden2 = dloss_func_dY_pred @ weights3.T * (hidden_layers2>0).astype(float)
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
        
        
    print(count)
    return weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,Y_pred,mu_Y,std_Y,loss,iter_hist,train_loss_hist,val_loss_hist

def carry_out(data,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=500000000,Max_patience=100000,patience=0.000000001,nodes=64):
    np.random.seed(42)
    np.random.shuffle(data)  # data 的行顺序被打乱

    data_train=data[:6000,:] #前6000行作为训练集
    data_foucus=data[6000:8000,:] #2000行作为验证集
    data_test=data[8000:,:] #后2000行作为测试集
    

    # Diagnose whether sequential split introduces train/test distribution shift.
    diagnose_split_distribution( data_test,data_foucus,data_train, top_k_features=4)

    weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,Y_train_pred,mu_Y,std_Y,loss,iter_hist,train_loss_hist,val_loss_hist=house_price_prediction_3layers_train(data_train,data_foucus,learning_rate, l2_lambda, Max_iterations, Max_patience, patience, nodes)
    Y_train_pred_original = reverse_standardize(Y_train_pred, mu_Y, std_Y)
    Y_train_pred_original=np.expm1(Y_train_pred_original)
    y_train_true = data_train[:,-1:]
    loss_list_y, RMSE, MAE, P90 = report_error_metrics(y_train_true, Y_train_pred_original, "Training")
    

    Y_pred,loss = house_price_prediction_3layers_predict(data_test,weights,offsets,weights2,offsets2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
    Y_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
    Y_pred_original=np.expm1(Y_pred_original)
    y_test_true = data_test[:,-1:]
    loss_list_ypred, RMSE, MAE, P90 = report_error_metrics(y_test_true, Y_pred_original, "Testing")
    
    return loss_list_y, loss_list_ypred, iter_hist, train_loss_hist, val_loss_hist, mu_X, std_X, mu_Y, std_Y
  

def main():
    data=readcsv('MLP_data.csv')
  
    np.random.seed(42)
    np.random.shuffle(data) # data 的行顺序被打乱

    loss_list_y, loss_list_ypred, iter_hist, train_loss_hist, val_loss_hist, mu_X, std_X, mu_Y, std_Y = carry_out(data,learning_rate=0.005,l2_lambda=0.00001,Max_iterations=600000,Max_patience=5,patience=0.00002,nodes=64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].scatter(np.arange(loss_list_y.shape[0]),loss_list_y.ravel(), color='blue', s=10,alpha=0.5)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Residual (Actual Price - Predicted Price)')
    axes[0].set_title('Training Error Scatter Plot')
    
    axes[1].scatter(np.arange(loss_list_ypred.shape[0]),loss_list_ypred.ravel(), color='red', s=10,alpha=0.5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Residual (Actual Price - Predicted Price)')
    axes[1].set_title('Testing Error Scatter Plot')

    axes[2].plot(iter_hist, train_loss_hist, color='blue', label='Training Loss')
    axes[2].plot(iter_hist, val_loss_hist, color='orange', label='Validation Loss')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss Curve')
    axes[2].grid(alpha=0.3)
    axes[2].legend()

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
    axes[2].text(
        0.02,
        0.98,
        process_note,
        transform=axes[2].transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    plt.tight_layout()
    plt.show()



   





if __name__ == "__main__":   
    main()