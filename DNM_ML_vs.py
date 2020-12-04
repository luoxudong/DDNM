# coding:utf-8
# 导入tensorflow。
# 这句import tensorflow as tf是导入TensorFlow约定俗成的做法，请大家记住。
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
tf.compat.v1.disable_eager_execution()
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC#引入SVM分类器
from sklearn.neighbors import KNeighborsClassifier   #引入KNN分类器
from scipy import stats
from sklearn.model_selection import KFold
import matlab.engine
import time
import math
eng = matlab.engine.start_matlab()


def get_p_value(arrA, arrB):
    a = np.array(arrA,dtype='float32')
    b = np.array(arrB,dtype='float32')
    res1 = stats.wilcoxon(b, a, alternative='greater')
    w_1 = res1.statistic;
    res = stats.wilcoxon(a, b, alternative='greater')
    w_2 = res.statistic;
    pvalue = res.pvalue;
    return w_2,w_1,pvalue
parameter = {};
name = 'Cancer_dataset'#Vertebral_Column Transfusion heart Glass Ionosphere parkinsons
data_file_encode = "gb18030"
data_ex = pd.read_excel(name+'.xlsx')
data = data_ex.values;


#hyperparameters  k ks qs C Gamma K-value MLS M
parameter['Cancer_dataset'] = [9, 10, 0.5, 0.8, 0.5, 1, 3,20];
parameter['Vertebral_Column'] = [5, 15, 0.7, 1.0, 1.0, 9, 13,6];#
parameter['Transfusion'] = [5, 15, 0.7, 1.0, 1.0, 9, 13,8];
parameter['parkinsons'] = [5, 15, 0.8, 1.0, 1.0, 1, 1,9];
parameter['Ionosphere'] = [10, 17, 0.2, 0.8, 0.5, 2, 1,8];
parameter['heart'] = [4, 6, 0.6, 1.0, 0.8, 8, 3,16];
parameter['Glass'] = [4, 17, 0.7, 1.0, 0.9, 1, 1,8];
iter = 1000

#min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 这里feature_range根据需要自行设置，默认（0,1）
#for i in range(data.shape[1]):
    # Normalization
#    data[:, i:i+1] = min_max_scaler.fit_transform(data[:, i:i+1])

row = data.shape[1] #读取矩阵长度c
x_data = data[:, 0:row-1]
k1 = parameter[name][0] #dnm
k2 = parameter[name][1]#dnm
qs = parameter[name][2]#dnm
C_value = parameter[name][3] # svm
Gamma_value = parameter[name][4] # svm
K_value = parameter[name][5] # knn
minleafsize = parameter[name][6] # dt
M1 = parameter[name][7]   #dnm(2) bpnn(2)
t = -1
y_data = data[:, row-1:row]
temp1 = np.array(['rstate','Knn Train Accuracy', 'Knn Test Accuracy', 'svm Train Accuracy','svm Test Accuracy','dt Train Accuracy','dt Test Accuracy', 'bpnn Train Accuracy', 'bpnn Test Accuracy'])
temp = np.array(['m','rstate','Test Accuracy', 'Train Accuracy', 'AUC', 'Recall', 'F1', 'k1', 'k2'])
temp_orig = np.array(['m','rstate','Test Accuracy', 'Train Accuracy', 'AUC', 'Recall', 'F1', 'k1', 'k2'])
ACC_train_dnm = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_test_dnm = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_train_ddnm = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_test_ddnm = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_test_bpnn = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_train_bpnn = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

ACC_train_dnm1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_test_dnm1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

ACC_test_bpnn1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
ACC_train_bpnn1 = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for rstate in range(6):
    kf = KFold(n_splits = 5, shuffle = True)
    for train_index, test_index in kf.split(x_data):
        X_train, X_test, y_train, y_test = x_data[train_index], x_data[test_index], y_data[train_index], y_data[test_index]
        standardScaler = MinMaxScaler()
        standardScaler.fit(x_data)
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)
        X_train= X_train.tolist()
        y_train= y_train.tolist()
        X_test = X_test.tolist()
        y_test = y_test.tolist()
        #调用matlab中的决策树 生成w 及 q 并计算出决策树的 testing Accuracy及trainingAccuracy #代码已经加密如果需要请发邮件：luoxudong@gxnu.edu.cn
        [ww, qq, accurate_dt_train, accurate_dt_test]= eng.DT_matlab(matlab.double(X_train),matlab.double(y_train), matlab.double(X_test), matlab.double(y_test),minleafsize,nargout=4)
        #根据w获取需要的树突层数量
        M = np.array(ww).shape[0]
        x = tf.compat.v1.placeholder(tf.float32, [None, row-1])
        y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
        W = tf.Variable(np.array(ww),dtype=tf.float32)
        q = tf.Variable(np.array(qq),dtype=tf.float32)
        t += 1
        X = tf.expand_dims(x, 1)
        X = tf.tile(X, [1, M, 1])



        y_temp = tf.nn.sigmoid(k1 * 1.0 * (tf.multiply(X, W) - q))
        y = tf.reduce_prod((y_temp), axis=2)
        y1 = (tf.reduce_sum(y, axis=1, keepdims=True))
        y2 = tf.nn.sigmoid(k2*1.0*(y1-qs))
        LOSS = []

        output = []
        output1 = []
        loss = tf.reduce_mean(0.5 * tf.square(y_ - y2))
        # 有了损失，我们就可以用随机梯度下降针对模型的参数（W和b）进行优化
        train_step = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)


        with tf.compat.v1.Session() as sess:
            # 变量初始化
            sess.run(tf.compat.v1.global_variables_initializer())
            for _ in range(iter):
                sess.run(train_step, feed_dict={x: X_train, y_: y_train})
                LOSS.append(sess.run(loss, feed_dict={x: X_train, y_: y_train}))
                ddnm_accuracy_test_value = sess.run(y2, feed_dict={x: X_test})
                ddnm_accuracy_test_value = np.int64(ddnm_accuracy_test_value > 0.5)
                ddnm_accuracy_test = accuracy_score(y_test, ddnm_accuracy_test_value)
                ACC_test_ddnm[t].append(ddnm_accuracy_test)
                ddnm_accuracy_train_value = sess.run(y2, feed_dict={x: X_train})
                ddnm_accuracy_train_value = np.int64(ddnm_accuracy_train_value > 0.5)
                ddnm_accuracy_train = accuracy_score(y_train, ddnm_accuracy_train_value)
                ACC_train_ddnm[t].append(ddnm_accuracy_train)


            prediction_value = sess.run(y2, feed_dict={x: X_test})
            prediction_value = np.int64(prediction_value > 0.5)
            test_accuracy = accuracy_score(y_test, prediction_value)
            print("dnm test accuracy:",accuracy_score(y_test, prediction_value))


            prediction_value2 = sess.run(y2, feed_dict={x: X_train})
            prediction_value2 = np.int64(prediction_value2 > 0.5)
            train_accuracy = accuracy_score(y_train, prediction_value2)

            test_fpr, test_tpr, test_threshold = roc_curve(y_test, prediction_value)  ###计算真正率和假正率
            test_auc = auc(test_fpr, test_tpr)

            test_recall = recall_score(y_test, prediction_value)

            test_f1 = f1_score(y_test, prediction_value)

            p_value = precision_score(y_test, prediction_value)
            output.append(M)
            output.append(rstate)
            output.append(test_accuracy)
            output.append(train_accuracy)
            output.append(test_auc)
            output.append(test_recall)
            output.append(test_f1)
            output.append(k1)
            output.append(k2)
            nparray1 = np.array(output)
            temp = np.row_stack((temp, nparray1))

        x_orig = tf.compat.v1.placeholder(tf.float32, [None, row - 1])
        y__orig = tf.compat.v1.placeholder(tf.float32, [None, 1])
        W_orig = tf.Variable(tf.compat.v1.random_uniform(shape = [M1, row-1], minval=-1, maxval=1, dtype=tf.float32),name='W_orig')
        q_orig = tf.Variable(tf.compat.v1.random_uniform(shape = [M1, row-1], minval=-1, maxval=1, dtype=tf.float32),name='q_orig')


        X_orig = tf.expand_dims(x_orig, 1)
        X_orig = tf.tile(X_orig, [1, M1, 1])
        fig = plt.figure()
        y_temp_orig = tf.nn.sigmoid(k1 * 1.0 * (tf.multiply(X_orig, W_orig) - q_orig))
        y_orig = tf.reduce_prod((y_temp_orig), axis=2)
        y1_orig = (tf.reduce_sum(y_orig, axis=1, keepdims=True))
        y2_orig = tf.nn.sigmoid(k2 * 1.0 * (y1_orig - 0.5))
        LOSS_orig = []
        output_orig = []
        output1_orig = []
        loss_orig = tf.reduce_mean(0.5 * tf.square(y__orig - y2_orig))
        train_step_orig = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_orig)
        start = time.time()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for _ in range(iter):
                sess.run(train_step_orig, feed_dict={x_orig: X_train, y__orig: y_train})
                LOSS_orig.append(sess.run(loss_orig, feed_dict={x_orig: X_train, y__orig: y_train}))

                dnm_accuracy_test_value = sess.run(y2_orig, feed_dict={x_orig: X_test})
                dnm_accuracy_test_value = np.int64(dnm_accuracy_test_value > 0.5)
                dnm_accuracy_test = accuracy_score(y_test, dnm_accuracy_test_value)
                ACC_test_dnm[t].append(dnm_accuracy_test)
                dnm_accuracy_train_value = sess.run(y2_orig, feed_dict={x_orig: X_train})
                dnm_accuracy_train_value = np.int64(dnm_accuracy_train_value > 0.5)
                dnm_accuracy_train = accuracy_score(y_train, dnm_accuracy_train_value)
                ACC_train_dnm[t].append(dnm_accuracy_train)


            prediction_value = sess.run(y2_orig, feed_dict={x_orig: X_test})
            prediction_value = np.int64(prediction_value > 0.5)

            test_accuracy = accuracy_score(y_test, prediction_value)
            print("dnm test accuracy:", accuracy_score(y_test, prediction_value))

            prediction_value2 = sess.run(y2_orig, feed_dict={x_orig: X_train})
            prediction_value2 = np.int64(prediction_value2 > 0.5)

            train_accuracy = accuracy_score(y_train, prediction_value2)

            test_fpr, test_tpr, test_threshold = roc_curve(y_test, prediction_value)  ###计算真正率和假正率
            test_auc = auc(test_fpr, test_tpr)

            test_recall = recall_score(y_test, prediction_value)

            test_f1 = f1_score(y_test, prediction_value)

            p_value = precision_score(y_test, prediction_value)

            output_orig.append(M1)
            output_orig.append(rstate)
            output_orig.append(test_accuracy)
            output_orig.append(train_accuracy)
            output_orig.append(test_auc)
            output_orig.append(test_recall)
            output_orig.append(test_f1)
            output_orig.append(k1)
            output_orig.append(k2)

            nparray_orig = np.array(output_orig)


            temp_orig = np.row_stack((temp_orig, nparray_orig))
        LOSS_bpnn = []
        n_hidden_1 = math.floor(M1 * 2 * (row - 1) / (row + 1))
        x_bpnn = tf.compat.v1.placeholder(tf.float32, [None, row - 1])
        y__bpnn = tf.compat.v1.placeholder(tf.float32, [None, 1])
        w1_bpnn = tf.Variable(tf.compat.v1.random_normal([row - 1, n_hidden_1]))
        q1_bpnn = tf.Variable(tf.compat.v1.random_normal([n_hidden_1]))
        w2_bpnn = tf.Variable(tf.compat.v1.random_normal([n_hidden_1, 1]))
        q2_bpnn = tf.Variable(tf.compat.v1.random_normal([1]))
        layer1 = tf.sigmoid(tf.add(tf.matmul(x_bpnn, w1_bpnn), q1_bpnn))
        y2_bpnn = tf.matmul(layer1, w2_bpnn) + q2_bpnn
        loss_bpnn = tf.reduce_mean(0.5 * tf.square(y__bpnn - y2_bpnn))
        train_step_bpnn = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_bpnn)


        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for _ in range(iter):
                sess.run(train_step_bpnn, feed_dict={x_bpnn: X_train, y__bpnn: y_train})
                LOSS_bpnn.append(sess.run(loss_bpnn, feed_dict={x_bpnn: X_train, y__bpnn: y_train}))

                bpnn_accuracy_test_value = sess.run(y2_bpnn, feed_dict={x_bpnn: X_test})
                bpnn_accuracy_test_value = np.int64(bpnn_accuracy_test_value > 0.5)
                bpnn_accuracy_test = accuracy_score(y_test, bpnn_accuracy_test_value)
                ACC_test_bpnn[t].append(bpnn_accuracy_test)
                bpnn_accuracy_train_value = sess.run(y2_bpnn, feed_dict={x_bpnn: X_train})
                bpnn_accuracy_train_value = np.int64(bpnn_accuracy_train_value > 0.5)
                bpnn_accuracy_train = accuracy_score(y_train, bpnn_accuracy_train_value)
                ACC_train_bpnn[t].append(bpnn_accuracy_train)

            prediction_value = sess.run(y2_bpnn, feed_dict={x_bpnn: X_test})
            prediction_value = np.int64(prediction_value > 0.5)

            accurate_bpnn_test = accuracy_score(y_test, prediction_value)
            print("dnm test accuracy:", accuracy_score(y_test, prediction_value))

            prediction_value2 = sess.run(y2_bpnn, feed_dict={x_bpnn: X_train})
            prediction_value2 = np.int64(prediction_value2 > 0.5)

            accurate_bpnn_train = accuracy_score(y_train, prediction_value2)
        M = np.array(ww).shape[0]
        x_orig1 = tf.compat.v1.placeholder(tf.float32, [None, row - 1])
        y__orig1 = tf.compat.v1.placeholder(tf.float32, [None, 1])
        W_orig1 = tf.Variable(tf.compat.v1.random_uniform(shape=[M, row - 1], minval=-1, maxval=1, dtype=tf.float32),
                              name='W_orig1')
        q_orig1 = tf.Variable(tf.compat.v1.random_uniform(shape=[M, row - 1], minval=-1, maxval=1, dtype=tf.float32),
                              name='q_orig1')

        X_orig1 = tf.expand_dims(x_orig1, 1)
        X_orig1 = tf.tile(X_orig1, [1, M, 1])
        fig = plt.figure()
        y_temp_orig1 = tf.nn.sigmoid(k1 * 1.0 * (tf.multiply(X_orig1, W_orig1) - q_orig1))
        y_orig1 = tf.reduce_prod((y_temp_orig1), axis=2)
        y1_orig1 = (tf.reduce_sum(y_orig1, axis=1, keepdims=True))
        y2_orig1 = tf.nn.sigmoid(k2 * 1.0 * (y1_orig1 - 0.5))
        LOSS_orig1 = []
        output_orig1 = []
        output1_orig1 = []
        loss_orig1 = tf.reduce_mean(0.5 * tf.square(y__orig1 - y2_orig1))
        train_step_orig1 = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_orig1)
        start = time.time()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for _ in range(iter):
                sess.run(train_step_orig1, feed_dict={x_orig1: X_train, y__orig1: y_train})
                LOSS_orig1.append(sess.run(loss_orig1, feed_dict={x_orig1: X_train, y__orig1: y_train}))

                dnm_accuracy_test_value = sess.run(y2_orig1, feed_dict={x_orig1: X_test})
                dnm_accuracy_test_value = np.int64(dnm_accuracy_test_value > 0.5)
                dnm_accuracy_test = accuracy_score(y_test, dnm_accuracy_test_value)
                ACC_test_dnm1[t].append(dnm_accuracy_test)
                dnm_accuracy_train_value = sess.run(y2_orig1, feed_dict={x_orig1: X_train})
                dnm_accuracy_train_value = np.int64(dnm_accuracy_train_value > 0.5)
                dnm_accuracy_train = accuracy_score(y_train, dnm_accuracy_train_value)
                ACC_train_dnm1[t].append(dnm_accuracy_train)

            prediction_value = sess.run(y2_orig1, feed_dict={x_orig1: X_test})
            prediction_value = np.int64(prediction_value > 0.5)

            test_accuracy = accuracy_score(y_test, prediction_value)
            print("dnm test accuracy:", accuracy_score(y_test, prediction_value))

            prediction_value2 = sess.run(y2_orig1, feed_dict={x_orig1: X_train})
            prediction_value2 = np.int64(prediction_value2 > 0.5)

            train_accuracy = accuracy_score(y_train, prediction_value2)

            test_fpr, test_tpr, test_threshold = roc_curve(y_test, prediction_value)  ###计算真正率和假正率
            test_auc = auc(test_fpr, test_tpr)

            test_recall = recall_score(y_test, prediction_value)

            test_f1 = f1_score(y_test, prediction_value)

            p_value = precision_score(y_test, prediction_value)


        LOSS_bpnn1 = []
        n_hidden_1 = math.floor(M * 2 * (row - 1) / (row + 1))
        x_bpnn1 = tf.compat.v1.placeholder(tf.float32, [None, row - 1])
        y__bpnn1 = tf.compat.v1.placeholder(tf.float32, [None, 1])
        w1_bpnn1 = tf.Variable(tf.compat.v1.random_normal([row - 1, n_hidden_1]))
        q1_bpnn1 = tf.Variable(tf.compat.v1.random_normal([n_hidden_1]))
        w2_bpnn1 = tf.Variable(tf.compat.v1.random_normal([n_hidden_1, 1]))
        q2_bpnn1 = tf.Variable(tf.compat.v1.random_normal([1]))
        layer1 = tf.sigmoid(tf.add(tf.matmul(x_bpnn1, w1_bpnn1), q1_bpnn1))
        y2_bpnn1 = tf.matmul(layer1, w2_bpnn1) + q2_bpnn1
        loss_bpnn1 = tf.reduce_mean(0.5 * tf.square(y__bpnn1 - y2_bpnn1))
        train_step_bpnn1 = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss_bpnn1)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for _ in range(iter):
                sess.run(train_step_bpnn1, feed_dict={x_bpnn1: X_train, y__bpnn1: y_train})
                LOSS_bpnn1.append(sess.run(loss_bpnn1, feed_dict={x_bpnn1: X_train, y__bpnn1: y_train}))

                bpnn1_accuracy_test_value = sess.run(y2_bpnn1, feed_dict={x_bpnn1: X_test})
                bpnn1_accuracy_test_value = np.int64(bpnn1_accuracy_test_value > 0.5)
                bpnn1_accuracy_test = accuracy_score(y_test, bpnn1_accuracy_test_value)
                ACC_test_bpnn1[t].append(bpnn1_accuracy_test)
                bpnn1_accuracy_train_value = sess.run(y2_bpnn1, feed_dict={x_bpnn1: X_train})
                bpnn1_accuracy_train_value = np.int64(bpnn1_accuracy_train_value > 0.5)
                bpnn1_accuracy_train = accuracy_score(y_train, bpnn1_accuracy_train_value)
                ACC_train_bpnn1[t].append(bpnn1_accuracy_train)

            prediction_value = sess.run(y2_bpnn1, feed_dict={x_bpnn1: X_test})
            prediction_value = np.int64(prediction_value > 0.5)

            accurate_bpnn1_test = accuracy_score(y_test, prediction_value)
            print("dnm test accuracy:", accuracy_score(y_test, prediction_value))

            prediction_value2 = sess.run(y2_bpnn1, feed_dict={x_bpnn1: X_train})
            prediction_value2 = np.int64(prediction_value2 > 0.5)

            accurate_bpnn1_train = accuracy_score(y_train, prediction_value2)

        end = time.time()
        print(start - end)

        end = time.time()
        print(start - end)

        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(LOSS, lw=2, label='DDNM(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
        # ax.plot(LOSS_orig, 'r--', lw=2, label='DNM(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
        #
        # plt.legend()
        # plt.xlabel('Learning Epoch')
        # plt.ylabel('MSE')
        # plt.title('Data by DNM')
        #
        # plt.show()

        knn=KNeighborsClassifier(n_neighbors = K_value)#调用knn分类器
        knn.fit(X_train,y_train)#训练knn分类器
        accurate_Knn_train=knn.score(X_train,y_train,sample_weight=None)#调用该对象的打分方法，计算出准确率
        accurate_Knn_test=knn.score(X_test,y_test,sample_weight=None)#调用该对象的打分方法，计算出准确率

        print(rstate,'KNN输出训练集的准确率为：',accurate_Knn_train)
        print(rstate,'KNN输出测试集的准确率为：',accurate_Knn_test)


        print("\n\n")


        #2.SVM算法分类
        svm=SVC(kernel='rbf',gamma=Gamma_value,decision_function_shape='ovo',C=C_value)#搭建模型，训练SVM分类器
        svm.fit(X_train,y_train)#训练SVC


        accurate_Svm_train=svm.score(X_train,y_train,sample_weight=None)#调用该对象的打分方法，计算出准确率
        accurate_Svm_test=svm.score(X_test,y_test,sample_weight=None)#调用该对象的打分方法，计算出准确率


        print(rstate,'Svm输出训练集的准确率为：',accurate_Svm_train)
        print(rstate,'Svm输出测试集的准确率为：',accurate_Svm_test)

        print("\n\n")




        output1.append(rstate)
        output1.append(accurate_Knn_train)
        output1.append(accurate_Knn_test)

        output1.append(accurate_Svm_train)
        output1.append(accurate_Svm_test)

        output1.append(accurate_dt_train)
        output1.append(accurate_dt_test)

        output1.append(accurate_bpnn_train)
        output1.append(accurate_bpnn_test)
        nparray2 = np.array(output1)
        temp1 = np.row_stack((temp1, nparray2))
fig = plt.figure(1)
ax2 = fig.add_subplot(1, 1, 1)
        #len_acc = len(ACC_test_dnm)
        #inter = len_acc//20  # 间隔
ACC_test_dnm_mean = np.mean(ACC_test_dnm, axis=0)

ACC_train_dnm_mean = np.mean(ACC_train_dnm, axis=0)

ACC_test_dnm1_mean = np.mean(ACC_test_dnm1, axis=0)

ACC_train_dnm1_mean = np.mean(ACC_train_dnm1, axis=0)

ACC_test_ddnm_mean = np.mean(ACC_test_ddnm, axis=0)

ACC_train_ddnm_mean = np.mean(ACC_train_ddnm, axis=0)


ACC_test_bpnn_mean = np.mean(ACC_test_bpnn, axis=0)

ACC_train_bpnn_mean = np.mean(ACC_train_bpnn, axis=0)

ACC_test_bpnn1_mean = np.mean(ACC_test_bpnn1, axis=0)

ACC_train_bpnn1_mean = np.mean(ACC_train_bpnn1, axis=0)


ax2.grid()

ax2.plot(range(iter), ACC_test_dnm_mean, '--', color="r",
        label='DNM (2) test')
ax2.plot(range(iter), ACC_train_dnm_mean, '-', color="r",
        label='DNM (2) train')

ax2.plot(range(iter), ACC_test_dnm1_mean, '--', color="g",
        label='DNM (1) test')
ax2.plot(range(iter), ACC_train_dnm1_mean, '-', color="g",
        label='DNM (1) train')

ax2.plot(range(iter), ACC_test_ddnm_mean, '--', color="b",
        label='DDNM test')
ax2.plot(range(iter), ACC_train_ddnm_mean, '-', color="b",
        label='DDNM train')

ax2.plot(range(iter), ACC_test_bpnn_mean, '--', color="c",
        label='BPNN (2) test')
ax2.plot(range(iter), ACC_train_bpnn_mean, '-', color="c",
        label='BPNN (2) train')

ax2.plot(range(iter), ACC_test_bpnn1_mean, '--', color="m",
        label='BPNN (1) test')
ax2.plot(range(iter), ACC_train_bpnn1_mean, '-', color="m",
        label='BPNN (1) train')
        #x = np.arange(1, 20,len_acc)
#ax2.plot(ACC_test_dnm,'-', lw=1, label='DNM test(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(ACC_train_dnm, '-', lw=1, label='DNM train(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(ACC_test_ddnm, '-',lw=1, label='DDNM test(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(ACC_train_ddnm, '-', lw=1, label='DDNM train(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(x,ACC_test_dnm[0:len_acc:inter], 'o', lw=1, label='DNM test(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(x,ACC_train_dnm[0:len_acc:inter], '*', lw=1, label='DNM train(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(x,ACC_test_ddnm[0:len_acc:inter], 'o', lw=1, label='DDNM test(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
#ax2.plot(x,ACC_train_ddnm[0:len_acc:inter], '*', lw=1, label='DDNM train(k1 = %0.2f ,k2 = %0.2f)' % (k1, k2))  # r-表示红色，lw表示线的宽度
plt.legend(loc='lower right')
plt.xlabel('Learning Epoch')
plt.ylabel('Average Accuracy')
plt.title('')
plt.savefig('./'+name+'_acc.eps', format='eps')


plt.show()

data1 = pd.DataFrame(temp)
data1.to_csv(name+'result_DDNM.csv')

data_orig = pd.DataFrame(temp_orig)
data_orig.to_csv(name+'result_DNM_orig.csv')

data2 = pd.DataFrame(temp1)
data2.to_csv(name+'result_other_ML.csv')
dnmpkorigDNM = get_p_value(temp[1:,2],temp_orig[1:,2])
dnmpkknn = get_p_value(temp[1:,2],temp1[1:,2])
dnmpksvm = get_p_value(temp[1:,2],temp1[1:,4])
dnmpkdt = get_p_value(temp[1:,2],temp1[1:,6])
dnmpkbpnn = get_p_value(temp[1:,2],temp1[1:,8])

dnmpkorigDNM_train = get_p_value(temp[1:,3],temp_orig[1:,3])
dnmpkknn_train = get_p_value(temp[1:,3],temp1[1:,1])
dnmpksvm_train = get_p_value(temp[1:,3],temp1[1:,3])
dnmpkdt_train = get_p_value(temp[1:,3],temp1[1:,5])
dnmpkbpnn_train = get_p_value(temp[1:,3],temp1[1:,7])

print('ddnm mean accurate:',sum(np.array(temp[1:,2],dtype=float))/np.array(temp[1:,2],dtype=float).shape[0])
print('dnm_orig mean accurate:',sum(np.array(temp_orig[1:,2],dtype=float))/np.array(temp_orig[1:,2],dtype=float).shape[0])
print('knn mean accurate:',sum(np.array(temp1[1:,2],dtype=float))/np.array(temp1[1:,2],dtype=float).shape[0])
print('svm mean accurate:',sum(np.array(temp1[1:,4],dtype=float))/np.array(temp1[1:,4],dtype=float).shape[0])
print('dt mean accurate:',sum(np.array(temp1[1:,6],dtype=float))/np.array(temp1[1:,6],dtype=float).shape[0])
print('bpnn mean accurate:',sum(np.array(temp1[1:,8],dtype=float))/np.array(temp1[1:,8],dtype=float).shape[0])
print('ddnm vs dnm_orig p-value:',dnmpkorigDNM)
print('ddnm vs knn p-value:',dnmpkknn)
print('ddnm vs svm p-value:',dnmpksvm)
print('ddnm vs dt p-value:',dnmpkdt)
print('ddnm vs bpnn p-value:',dnmpkbpnn)

print('ddnm mean accurate:',sum(np.array(temp[1:,3],dtype=float))/np.array(temp[1:,3],dtype=float).shape[0])
print('dnm_orig mean accurate:',sum(np.array(temp_orig[1:,3],dtype=float))/np.array(temp_orig[1:,3],dtype=float).shape[0])
print('knn mean accurate:',sum(np.array(temp1[1:,1],dtype=float))/np.array(temp1[1:,1],dtype=float).shape[0])
print('svm mean accurate:',sum(np.array(temp1[1:,3],dtype=float))/np.array(temp1[1:,3],dtype=float).shape[0])
print('dt mean accurate:',sum(np.array(temp1[1:,5],dtype=float))/np.array(temp1[1:,5],dtype=float).shape[0])
print('bpnn mean accurate:',sum(np.array(temp1[1:,7],dtype=float))/np.array(temp1[1:,7],dtype=float).shape[0])

print('ddnm vs dnm_orig p-value:',dnmpkorigDNM_train)
print('ddnm vs knn p-value:',dnmpkknn_train)
print('ddnm vs svm p-value:',dnmpksvm_train)
print('ddnm vs dt p-value:',dnmpkdt_train)
print('ddnm vs bpnn p-value:',dnmpkbpnn_train)