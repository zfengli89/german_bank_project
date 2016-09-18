#-*- coding:utf-8 -*-
#代码主要用来调节参数、观察结果，未工程结构化，凑合看看。 有疑问可发邮件：lizhifeng@sugo.io

from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import chisquare
from scipy.stats import ttest_ind
from sklearn.linear_model import  LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.externals.six import StringIO
import pydot

#load原始数据
def load_data():
    print '*** load data ***'
    dataset = np.loadtxt('german_credit.csv', delimiter=',')
    print 'dataset:',dataset
    return  dataset

#提取key值
def find_key(arry):
    print '*** find key ***'
    keys = []
    items = {}
    for i in arry:
        key = i
        items[key] = 0

    for item in items:
        key1 = item
        keys.append(key1)
    print 'key:',keys
    return keys

#eda分析
def eda_arry(arry):
    print '*** EDA ***'
    arry_size = arry.shape
    arry_high = arry_size[0]
    keys = find_key(arry)
    x = 0
    print u'标签/数量/占比率'
    for key in keys:
        key_num = len([x for x in arry if x == key])
        rate = float(key_num)/arry_high
        print int(key),key_num, rate

#箱线图
def plot_box(a1, a2, a3):
    plt.figure(1)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    plt.boxplot(a1, labels='x')
    ax1.set_xlabel('Credit Amount')
    plt.sca(ax1)
    plt.boxplot(a2, labels='y')
    ax2.set_xlabel('Age (years)')
    plt.sca(ax2)
    plt.boxplot(a3, labels='z')
    ax3.set_xlabel('Credit (month)')
    plt.sca(ax3)
    plt.show()

#原始矩阵预处理
def get_newData(data):
    print '***get_newDate***'
    arry_size = data.shape
    width = arry_size[1]
    higth = arry_size[0]
    for index in range(higth):
        arry = data[index,:]
        #print arry
        #print index, arry
        #account balance
        if arry[1] == 4:
            arry[1] = 3
        #payment status
        if arry[3] == 1:
            arry[3] = 0
        if arry[3] == 2:
            arry[3] = 1
        if arry[3] == 3 or arry[3] == 4:
            arry[3] = 2
        #purpose
        if arry[4] == 4 or arry[4] ==5 or arry[4] ==6:
            arry[4] = 3
        if arry[4] == 8 or arry[4] ==9 or arry[4] ==10:
            arry[4] = 0
        #savings
        if arry[6] == 4:
            arry[6] = 3
        if arry[6] == 5:
            arry[6] = 4
        #length employment
        if arry[7] == 1:
            arry[7] = 2
        #sex
        if arry[9]  == 1:
            arry[9] = 2
        #guarantor
        if arry[10] == 3:
            arry[10] = 2
        #concurrent
        if arry[14] == 1:
            arry[14] = 2
        if arry[16] == 3 or arry[16] == 4:
            arry[16] = 2
        if arry[17] == 1:
            arry[17] = 2
        if index == 0:
            mat_new = arry
        else:
            mat_new = np.vstack((mat_new, arry))
    print mat_new
    return mat_new

#display矩阵详细报告
def display_data(data):
    arry_size = data.shape
    width = arry_size[1]
    higth = arry_size[0]
    print u'矩阵维度：M * N:', higth, width, '\n'

    for index in range(width):
        print u'特征：', index
        arry = data[:, index]
        eda_arry(arry)
        print '\n'

#分离数据
def sep_bad_good(data):
    print '***separa bad good mat***'
    arry_size = data.shape
    width = arry_size[1]
    higth = arry_size[0]
    mat_good =[]
    mat_bad = []
    for index in range(higth):
        arry = data[index, :]
        if arry[0] == 0:
            if len(mat_bad) == 0:
                mat_bad = arry
            else:
                mat_bad = np.vstack((mat_bad,arry))
        else:
            if len(mat_good) == 0:
                mat_good = arry
            else:
                mat_good = np.vstack((mat_good,arry))

    return mat_good, mat_bad

def list_sort(item):
    l = []
    for k in item:
        l.append({"name": k, "count": item[k]})

    l = sorted(l, key=lambda item1: item1["count"], reverse=True)
    print 'sort:',l

#皮尔逊检验
def test_pearson(data):
    print '*** pearson ***'
    X = data[:,:]
    y = data[:,0]
    print X.shape, y.shape
    mat_size = X.shape
    width = mat_size[1]
    print width
    item ={}
    for index in range(width):
        pearson_value, p = pearsonr(X[:,index], y)
        print index , pearson_value
        item[index] = abs(pearson_value)

    print item
    list_sort(item)

#卡方检验
def kf_test(data):
    print '*** KA--F ***'
    mat_size = data.shape
    higth = mat_size[0]
    width = mat_size[1]
    print 'h*w',higth, width
    good_mat, bad_mat = sep_bad_good(data)
    print good_mat.shape, bad_mat.shape
    item = {}
    for index in range(width):
        yuan_obj = []
        expect_obj = []
        arry_all = data[:,index]
        keys = find_key(arry_all)
        arry_good = good_mat[:,index]
        arry_bad =  bad_mat[:,index]
        size_good = arry_good.shape
        size_bad =  arry_bad.shape
        for key in keys:
            good_key_num = len([x for x in arry_good if x == key])
            bad_key_num  = len([x for x in arry_bad if x == key])
            yuan_obj.append(good_key_num)
            yuan_obj.append(bad_key_num)

            all_num = good_key_num + bad_key_num
            #print 'all num:',good_key_num, bad_key_num,all_num
            exp_good_num = all_num*0.70
            exp_bad_num = all_num*0.30
            expect_obj.append(exp_good_num)
            expect_obj.append(exp_bad_num)
        #print index, yuan_obj, expect_obj
        chisq, p = chisquare(yuan_obj, expect_obj)
        print 'kafang:',index, chisq, p, '\n'
        item[index] = abs(chisq)

    list_sort(item)

#t检验
def t_test(data):
    print '***t test***'
    good_mat, bad_mat = sep_bad_good(data)

    sta1, p1 = ttest_ind(good_mat[:,2], bad_mat[:,2])
    sta2, p2 = ttest_ind(good_mat[:,5], bad_mat[:,5])
    sta3, p3 = ttest_ind(good_mat[:,13], bad_mat[:,13])

    print 't p-value:', sta1,p1, sta2,p2, sta3, p3

#随机森林
def feature_select(data):
    print 'feature select:'
    X = data[:,1:]
    y = data[:,0]
    print X.shape, y.shape
    mat_size = X.shape
    width = mat_size[1]
    print width
    print X.shape
    model = ExtraTreesClassifier()
    print model
    model.fit(X,y)
    print 'feature_import', model.feature_importances_
    print_sortIndex(model.feature_importances_)

def print_sortIndex(arry):
    print np.argsort(-arry)


#load原始数据

def sample_test_train(data, flag):

    print '*** matrix random sample***'

    good_mat, bad_mat = sep_bad_good(data)

    good_size = good_mat.shape
    bad_size  = bad_mat.shape
    good_h = good_size[0]
    bad_h  = bad_size[0]
    print 'good_h , bad_h:', good_h, bad_h
    good_sample_cnt = good_h/2
    bad_sample_cnt  = bad_h/2

    print 'sample cnt, good/bad:', good_sample_cnt, bad_sample_cnt
    np.random.shuffle(good_mat)
    np.random.shuffle(bad_mat)

    good_train = good_mat[:int(good_sample_cnt)]
    good_test  = good_mat[int(good_sample_cnt):]

    bad_train = bad_mat[:int(bad_sample_cnt)]
    bad_test  = bad_mat[int(bad_sample_cnt):]

    train_data = np.vstack( (good_train,bad_train) )
    test_data  = np.vstack( (good_test, bad_test) )


    #选择部分特征
    print '使用好特征'
    train_mat = array( [ train_data[:,0], train_data[:,1], train_data[:,2], train_data[:,3],  train_data[:,4], train_data[:,5], train_data[:,6], train_data[:,7], train_data[:,11], train_data[:,12], train_data[:,13], train_data[:,14]])
    test_mat = array( [test_data[:, 0], test_data[:, 1], test_data[:, 2], test_data[:, 3], test_data[:, 4], test_data[:, 5], test_data[:, 6], test_data[:, 7], test_data[:, 11], test_data[:, 12], test_data[:, 13], test_data[:, 14],])

    # print '使用差特征'
    # train_mat = array( [ train_data[:,0], train_data[:,8], train_data[:,9], train_data[:,10], train_data[:,15],  train_data[:,16], train_data[:,17], train_data[:,18], train_data[:,19], train_data[:,20]])
    # test_mat = array( [test_data[:, 0], test_data[:, 8], test_data[:, 9], test_data[:, 10], test_data[:, 15], test_data[:, 16], test_data[:, 17], test_data[:, 18], test_data[:, 19], test_data[:, 20]])

    print 'train_mat:',train_mat

    last_train_mat = transpose(train_mat)
    last_test_mat = transpose(test_mat)

    if flag == 1:
        print u'不筛选特征：'
        train_X = train_data[:,1:]
        train_y = train_data[:,0]

        test_X = test_data[:,1:]
        test_y = test_data[:,0]
    else:
        print u'筛选特性后：'
        train_X = last_train_mat[:,1:]
        train_y = last_train_mat[:,0]

        test_X = last_test_mat[:,1:]
        test_y = last_test_mat[:,0]

    #logic_test(train_X, train_y, test_X, test_y)
    return train_X, train_y, test_X, test_y

#标准化
def standard_data(X):
    #normalized_X = preprocessing.normalize(X)
    #print 'normalized', normalized_X
    standardized_X = preprocessing.scale(X)
    print 'standardized', standardized_X
    return standardized_X

#逻辑回归模型
def logic_test(train_X, train_y, test_X, test_y):
    print 'logist regression ...'

    #特征规范法
    scaler = preprocessing.StandardScaler().fit(train_X)    #规范化对象
    print train_X.shape, test_X.shape
    stand_train_X = scaler.transform(train_X)
    stand_test_X = scaler.transform(test_X)

    #特征多项式示例化
    quadratic_featurizer = PolynomialFeatures(degree=1)  # 实例化一个二次多项式特征实例 degree代表二项式的幂
    ploy_train_X = quadratic_featurizer.fit_transform(stand_train_X)  # 用二次多项式对样本X值做变换
    ploy_test_X  = quadratic_featurizer.fit_transform(stand_test_X)

    #建立逻辑回归模型
    model = LogisticRegression(class_weight = 'balanced')
    model.fit(ploy_train_X,train_y)
    print 'model coef:',model.coef_
    print model

    # 观察结果
    expected = train_y
    predicted = model.predict(ploy_train_X)
    print '***训练集的结果：***'
    print (metrics.classification_report(expected, predicted))
    print ( metrics.confusion_matrix( expected, predicted))

    print '***测试集的结果：***'
    test_exp = test_y
    test_pred = model.predict(ploy_test_X)
    print (metrics.classification_report(test_exp, test_pred))
    print ( metrics.confusion_matrix( test_exp, test_pred))

    #print '训练集混淆矩阵'
    #cfu_mat = metrics.confusion_matrix(expected, predicted)
    print '测试集混淆矩阵'
    cfu_mat = metrics.confusion_matrix(test_exp, test_pred)

    rate_neg = float(cfu_mat[0][0]) / (cfu_mat[0][0]+cfu_mat[0][1])
    rate_pos = float(cfu_mat[1][1]) / (cfu_mat[1][0]+cfu_mat[1][1])
    rate_total = float(cfu_mat[0][0] + cfu_mat[1][1]) / (cfu_mat[0][0]+cfu_mat[0][1]+cfu_mat[1][0]+cfu_mat[1][1])
    return rate_neg, rate_pos, rate_total

#决策树
def my_tree(train_X, train_y, test_X, test_y):
    print '*** tree ***'
    model = tree.DecisionTreeClassifier()  # 算法模型
    model = model.fit(train_X, train_y)  # 模型训练
    dot_data = StringIO()
    tree.export_graphviz(model, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("iris.pdf")  # 写入pdf


    test_exp = test_y
    test_pred = model.predict(test_X)
    print (metrics.classification_report(test_exp, test_pred))
    print ( metrics.confusion_matrix( test_exp, test_pred))

    cfu_mat = metrics.confusion_matrix(test_exp, test_pred)

    rate_neg = float(cfu_mat[0][0]) / (cfu_mat[0][0]+cfu_mat[0][1])
    rate_pos = float(cfu_mat[1][1]) / (cfu_mat[1][0]+cfu_mat[1][1])
    rate_total = float(cfu_mat[0][0] + cfu_mat[1][1]) / (cfu_mat[0][0]+cfu_mat[0][1]+cfu_mat[1][0]+cfu_mat[1][1])
    return rate_neg, rate_pos, rate_total


#随机森林
def my_forest(train_X, train_y, test_X, test_y):
    print '*** forest ***'
    model = RandomForestClassifier(n_estimators = 8)  # 算法模型
    model = model.fit(train_X, train_y)  # 模型训练

    test_exp = test_y
    test_pred = model.predict(test_X)
    print (metrics.classification_report(test_exp, test_pred))
    print ( metrics.confusion_matrix( test_exp, test_pred))

    cfu_mat = metrics.confusion_matrix(test_exp, test_pred)

    rate_neg = float(cfu_mat[0][0]) / (cfu_mat[0][0]+cfu_mat[0][1])
    rate_pos = float(cfu_mat[1][1]) / (cfu_mat[1][0]+cfu_mat[1][1])
    rate_total = float(cfu_mat[0][0] + cfu_mat[1][1]) / (cfu_mat[0][0]+cfu_mat[0][1]+cfu_mat[1][0]+cfu_mat[1][1])
    return rate_neg, rate_pos, rate_total


#测试程序

#1 加载数据
data = load_data()

#分析原始数据
#eda_arry(arry)
# arry_size = data.shape
# width = arry_size[1]
# higth = arry_size[0]
# print u'矩阵维度：M * N:', higth, width, '\n'
#
# for index in range(width):
#     print u'特征：',index
#     arry = data[:,index]
#     eda_arry(arry)
#     print '\n'
#箱线图
#plot_box(data[:,2], data[:,5], data[:,13])

#2 修改原始数据（合并部分level）
new_mat = get_newData(data)

#display_data(new_mat)
#good_mat, bad_mat= sep_bad_good(data)
# print bad_mat.shape
# display_data(bad_mat)


#3 计算pearson系数
#test_pearson(new_mat)

#4 计算卡方
#kf_test(new_mat)

#5 t检验
#t_test(new_mat)

#6 随机森林
#feature_select(new_mat)

#7 逻辑回归
# list_bad = []
# list_good = []
# list_total = []
#
# for index in range(20):
#     print index
#     train_X, train_y, test_X, test_y = sample_test_train(new_mat, 0)  # 0筛选特征  1：不筛选特征
#     bad,good, total = logic_test(train_X, train_y, test_X, test_y)
#     list_bad.append(bad)
#     list_good.append(good)
#     list_total.append(total)
#
# print 'bad:',   mean(list_bad),   list_bad
# print 'good:',  mean(list_good),  list_good
# print 'total:', mean(list_total), list_total

#8 决策树
# list_bad = []
# list_good = []
# list_total = []
#
# for index in range(20):
#     print index
#     train_X, train_y, test_X, test_y = sample_test_train(new_mat, 0)  # 0筛选特征  1：不筛选特征
#     bad, good, total = my_tree(train_X, train_y, test_X, test_y)
#     list_bad.append(bad)
#     list_good.append(good)
#     list_total.append(total)
#
# print 'bad:',   mean(list_bad),   list_bad
# print 'good:',  mean(list_good),  list_good
# print 'total:', mean(list_total), list_total


#9 随机森林
list_bad = []
list_good = []
list_total = []

for index in range(20):
    print index
    train_X, train_y, test_X, test_y = sample_test_train(new_mat, 1)  # 0筛选特征  1：不筛选特征
    bad, good, total = my_forest(train_X, train_y, test_X, test_y)
    list_bad.append(bad)
    list_good.append(good)
    list_total.append(total)

print 'bad:',   mean(list_bad),   list_bad
print 'good:',  mean(list_good),  list_good
print 'total:', mean(list_total), list_total