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

def house_price_prediction_2layers_train(data):
    data = data.astype(float)
    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    
    # 标准化输入数据
    X, mu_X, std_X = standardize(X)
    # 标准化目标变量
    Y, mu_Y, std_Y = standardize(Y)
    
    #隐藏层节点一
    weights1=np.random.rand(X.shape[1],1) #4*1
    offset1 = 0.01

    #隐藏层节点二
    weights2=np.random.rand(X.shape[1],1) #4*1
    offset2 = 0.01
    #输出节点
    weights3= np.random.rand(2) #1*2
    offset3 = 0.01

    Y_pred = np.zeros(Y.shape)
    loss=100
    count=0
    Max_iterations=100000
    while loss>0.01 and count<Max_iterations:
     # 前向传播
        #隐藏层节点一
        hiddenlayer_matmul_s1=np.matmul(X,weights1)+offset1 #n*1
        hidden_layer_s1 = relu(hiddenlayer_matmul_s1)

        #隐藏层节点二
        hiddenlayer_matmul_s2=np.matmul(X,weights2)+offset2 #n*1
        hidden_layer_s2 = relu(hiddenlayer_matmul_s2)

        #输出节点
        Y_pred_s1=hidden_layer_s1*weights3[0]
        Y_pred_s2=hidden_layer_s2*weights3[1]
        z_out= Y_pred_s1+Y_pred_s2+offset3 #n*1
        Y_pred = liner(z_out) #n*1
        loss=loss_function(Y_pred,Y)
       # print(loss)
        # 反向传播
        #计算各个偏导数

        dloss_func_dY_pred=Y_pred-Y #n*1
        dY_pred_dz_out=1 #1
        dz_out_dY_pred_s1=1 
        dz_out_dY_pred_s2=1

        dY_pred_s1_dweights3_0=hidden_layer_s1 #n*1
        dY_pred_s2_dweights3_1=hidden_layer_s2 #n*1
        dY_pred_s1_dhidden_layer_s1=weights3[0] #1
        dY_pred_s2_dhidden_layer_s2=weights3[1] #1
        dhidden_layer_s1_drelu=(hiddenlayer_matmul_s1>0).astype(float) #n*1
        dhidden_layer_s2_drelu=(hiddenlayer_matmul_s2>0).astype(float) #n*1
        drelu_dweights1=X #n*4
        #更新权重和偏置
        learning_rate=0.01
        #更新权重隐藏层—>输出层
        weights3[0]=weights3[0]-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s1*dY_pred_s1_dweights3_0) 
        weights3[1]=weights3[1]-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s2*dY_pred_s2_dweights3_1) 
        
        #更新偏置隐藏层—>输出层 
        offset3=offset3-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out) 
        #更新权重输入层—>隐藏层
       

        weights1=weights1-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s1*dY_pred_s1_dhidden_layer_s1*dhidden_layer_s1_drelu*drelu_dweights1,axis=0).reshape(weights1.shape)
        weights2=weights2-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s2*dY_pred_s2_dhidden_layer_s2*dhidden_layer_s2_drelu*drelu_dweights1,axis=0).reshape(weights2.shape)
        #更新偏置输入层—>隐藏层 

        offset1=offset1-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s1*dY_pred_s1_dhidden_layer_s1*dhidden_layer_s1_drelu)
        offset2=offset2-learning_rate*np.mean(dloss_func_dY_pred*dY_pred_dz_out*dz_out_dY_pred_s2*dY_pred_s2_dhidden_layer_s2*dhidden_layer_s2_drelu)
        count+=1
    print(count)
    return weights1,offset1,weights2,offset2,weights3,offset3,mu_X,std_X,Y_pred,mu_Y,std_Y,loss

def house_price_prediction_2layers_predict(data,weights1,offset1,weights2,offset2,weights3,offset3,mu_X,std_X,mu_Y,std_Y):
    data = data.astype(float)
    X=data[:,:-1] #n*4
    Y=data[:,-1:]  #n*1
    
    # 标准化输入数据
    X = (X - mu_X) / (std_X + 1e-8)
    # 标准化目标变量
    Y = (Y - mu_Y) / (std_Y + 1e-8)
    
        #隐藏层节点一
    hiddenlayer_matmul_s1=np.matmul(X,weights1)+offset1 #n*1
    hidden_layer_s1 = relu(hiddenlayer_matmul_s1)

        #隐藏层节点二
    hiddenlayer_matmul_s2=np.matmul(X,weights2)+offset2 #n*1
    hidden_layer_s2 = relu(hiddenlayer_matmul_s2)

        #输出节点
    Y_pred_s1=hidden_layer_s1*weights3[0]
    Y_pred_s2=hidden_layer_s2*weights3[1]
    z_out= Y_pred_s1+Y_pred_s2+offset3 #n*1
    Y_pred = liner(z_out) #n*1
    return Y_pred
  
def main():
    data=readcsv('MLP_data.csv')
    data_train=data[:8000,:] #前8000行作为训练集
    weights1,offset1,weights2,offset2,weights3,offset3,mu_X,std_X,Y_train_pred,mu_Y,std_Y,loss=house_price_prediction_2layers_train(data_train)
    Y_train_pred_original = reverse_standardize(Y_train_pred, mu_Y, std_Y)
    loss_list_y=data_train[:,-1:]-Y_train_pred_original
    
   
    
    data_test=data[8000:,:] #后2000行作为测试集
    Y_pred = house_price_prediction_2layers_predict(data_test,weights1,offset1,weights2,offset2,weights3,offset3,mu_X,std_X,mu_Y,std_Y)
    Y_pred_original = reverse_standardize(Y_pred, mu_Y, std_Y)
    loss_list_ypred=data_test[:,-1:]-Y_pred_original
    
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