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

def diagnose_split_distribution(data_train, data_test, top_k_features=4):
    # Principle: compare train/test distribution statistics.
    # Large gaps indicate distribution shift caused by split strategy.
    x_train = data_train[:, :-1]
    x_test = data_test[:, :-1]
    y_train = data_train[:, -1]
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
    print("Target(y) train/test summary:")
    print(f"  train mean={y_train_s['mean']:.3f}, std={y_train_s['std']:.3f}, p10/p50/p90={y_train_s['p10']:.3f}/{y_train_s['p50']:.3f}/{y_train_s['p90']:.3f}")
    print(f"  test  mean={y_test_s['mean']:.3f}, std={y_test_s['std']:.3f}, p10/p50/p90={y_test_s['p10']:.3f}/{y_test_s['p50']:.3f}/{y_test_s['p90']:.3f}")

    mean_gap = abs(y_train_s['mean'] - y_test_s['mean']) / (abs(y_train_s['mean']) + 1e-8)
    std_gap = abs(y_train_s['std'] - y_test_s['std']) / (abs(y_train_s['std']) + 1e-8)
    print(f"Target relative gap: mean_gap={mean_gap:.3f}, std_gap={std_gap:.3f}")

    # Feature-level check: print top_k features with largest train/test mean gap.
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

def house_price_prediction_2layers_train(data):
    data = data.astype(float)
    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    Y=np.log1p(Y)
    
    # 标准化输入数据
    X, mu_X, std_X = standardize(X)
    # 标准化目标变量
    Y, mu_Y, std_Y = standardize(Y)
    learning_rate=0.05
   
    nodes=32
    weights=np.random.rand(4,nodes) #4*nodes
    offsets=np.full(nodes,0.01) #nodes
   
    #输出节点
    weights3= np.random.rand(nodes).reshape(nodes,1) #nodes*1
    offset3 = 0.01

    Y_pred = np.zeros(Y.shape)
    loss=100
    count=0
    Max_iterations=100000
    while loss>0.01 and count<Max_iterations:
     # 前向传播
     #隐藏层节点
        hidden_layers=np.matmul(X,weights)+offsets #n*nodes
        hidden_layers = relu(hidden_layers) #n*nodes
        #输出节点
        z_out = hidden_layers @ weights3 + offset3   # (n, 1)
        Y_pred = liner(z_out) #n*1   
        loss=loss_function(Y_pred,Y)
       # print(loss)
        # 反向传播
        #计算各个偏导数

        dloss_func_dY_pred=(Y_pred-Y)/Y.shape[0] #n*1
        dY_pred_dweights3=hidden_layers #n*nodes
        #dY_pred_dweights=weights3*(hidden_layers>0).astype(float)*X #nodes*1 n*nodes   n*1   目标4*nodes
        #dY_pred_doffsets=weights3*(hidden_layers>0).astype(float)

        #梯度
        T_weights3=dY_pred_dweights3.T @ dloss_func_dY_pred #n*nodes.T x n*1 按列平均得到1*32
        T_offsets3=np.sum(dloss_func_dY_pred) #1
        T_weights=X.T @ (dloss_func_dY_pred @ weights3.T * (hidden_layers>0).astype(float) ) #4*nodes 
        T_offsets=np.sum(dloss_func_dY_pred @ weights3.T * (hidden_layers>0).astype(float), axis=0) #nodes
        #更新权重和偏置
        weights3 -= learning_rate * T_weights3
        weights -= learning_rate * T_weights
        offsets -= learning_rate * T_offsets
        offset3 -= learning_rate * T_offsets3
      
        count+=1
    print(count)
    return weights,offsets,weights3,offset3,mu_X,std_X,Y_pred,mu_Y,std_Y,loss

def house_price_prediction_2layers_predict(data,weights,offsets,weights3,offset3,mu_X,std_X,mu_Y,std_Y):
    data = data.astype(float)
    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    Y=np.log1p(Y)
    nodes=32
    # 标准化输入数据
    X = (X - mu_X) / (std_X + 1e-8)
    # 标准化目标变量
    Y = (Y - mu_Y) / (std_Y + 1e-8)
    
     #隐藏层节点
    hidden_layers=np.matmul(X,weights)+offsets #n*nodes
    hidden_layers = relu(hidden_layers) #n*nodes
    #输出节点
    z_out = hidden_layers @ weights3 + offset3   # (n, 1)
    Y_pred = liner(z_out) #n*1   
    loss=loss_function(Y_pred,Y)
   

        #输出节点
  
    return Y_pred,loss
  
def main():
    data=readcsv('MLP_data.csv')
    np.random.shuffle(data)   # data 的行顺序被打乱
    data_train=data[:8000,:] #前8000行作为训练集
    data_test=data[8000:,:] #后2000行作为测试集

    # Diagnose whether sequential split introduces train/test distribution shift.
    diagnose_split_distribution(data_train, data_test, top_k_features=4)

    weights,offsets,weights3,offset3,mu_X,std_X,Y_train_pred,mu_Y,std_Y,loss=house_price_prediction_2layers_train(data_train)
    Y_train_pred_original = reverse_standardize(Y_train_pred, mu_Y, std_Y)
    Y_train_pred_original=np.expm1(Y_train_pred_original)
    y_train_true = data_train[:,-1:]
    loss_list_y, RMSE, MAE, P90 = report_error_metrics(y_train_true, Y_train_pred_original, "Training")
    print("Training Loss:", loss)

    Y_pred,loss = house_price_prediction_2layers_predict(data_test,weights,offsets,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
    Y_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
    Y_pred_original=np.expm1(Y_pred_original)
    y_test_true = data_test[:,-1:]
    loss_list_ypred, RMSE, MAE, P90 = report_error_metrics(y_test_true, Y_pred_original, "Testing")
    print("Testing Loss:", loss)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(np.arange(loss_list_y.shape[0]),loss_list_y.ravel(), color='blue', s=10,alpha=0.5)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Residual (Actual Price - Predicted Price)')
    axes[0].set_title('Training Error Scatter Plot')
    
    axes[1].scatter(np.arange(loss_list_ypred.shape[0]),loss_list_ypred.ravel(), color='red', s=10,alpha=0.5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Residual (Actual Price - Predicted Price)')
    axes[1].set_title('Testing Error Scatter Plot')
    plt.tight_layout() 
    plt.show()
if __name__ == "__main__":   
    main()