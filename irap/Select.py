# import packages
import os
import numpy as np
import pandas as pd
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import Load as iload
import Visual as ivis
import SVM as isvm
import Evaluate as ieval
import Plot as iplot
import math
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# PCA #########################################################################
# 特征组合测试
def select_test(data, label, feature, c_number, gamma, crossv, now_path):
    
    fs_acc = []
    filter_data = []
    for k in label:
        filter_data.append(str(k))
    start_e = 0
    for i in range(len(feature)):
        start_e += 1
        key = feature[i]
        for j in range(len(data)):
            filter_data[j] += ' ' + str(i + 1) + ':' + str(data[j,key])
        out_content = ''
        for n in filter_data:
            out_content += n + '\n'
        with open('mid-ifs', 'w') as ot:
            ot.write(out_content)
        test_label, predict_label = isvm.svm_evaluate(os.path.join(now_path, 'mid-ifs'),
                                                 float(c_number), float(gamma), int(crossv))
        standard_num = ieval.evaluate_score(test_label, predict_label)
        single_acc = round(standard_num[4], 3)
        fs_acc.append(single_acc)
        os.remove('./mid-ifs')
        ivis.visual_easy_time(start_e, len(feature))
    return fs_acc


def pca_scale(x_train):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(x_train)
    x_train_ = scaler.transform(x_train)
    return x_train_


# pca
def select_pca(data):
    x = data.iloc[:, :]
    x_s = pca_scale(x)
    pca = PCA(n_components=1)
    pca.fit(x_s)
    pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc1_featurescore = pd.DataFrame({'Feature': x.columns,
                                     'PC1_loading': pc1_loadings.T[0],
                                     'PC1_loading_abs': abs(pc1_loadings.T[0])})
    pc1_featurescore = pc1_featurescore.sort_values('PC1_loading_abs', ascending=False)
    feature_selection = []
    for i in pc1_featurescore['Feature']:
        feature_selection.append(i)
    return feature_selection


# save
def select_save(out, fs_sort):
    out_file = 'IFS-feature-sort: '
    for j in fs_sort:
        out_file += str(j + 1) + ' '
    with open(out, 'w', encoding='UTF-8') as f:
        f.write(out_file)
        f.close()


# select pca main for svm
def select_svm_pca(in_path, c=8, g=0.125, cv=5, out_path=now_path, all_p=True):
    if out_path != None:
        if os.path.split(out_path)[1] not in os.listdir(os.path.split(out_path)[0]):
            os.makedirs(out_path)
        # load svm
        np_data, np_label = iload.load_svmfile(in_path)
        # PCA
        pd_data = pd.DataFrame(np_data)
        fs_sort = select_pca(pd_data)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        print('\n特征筛选完成，导出结果中...')
        # plot
        if all_p == True:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-pca.png'), in_path=in_path)
        else:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-pca.png'))
        # save results
        select_save(os.path.join(out_path, 'Fsort-pca.txt'), fs_sort)
        return fs_acc.index(max(fs_acc))
    else:
        # 读取文件
        np_data, np_label = iload.load_svmfile(in_path)
        # PCA
        pd_data = pd.DataFrame(np_data)
        fs_sort= select_pca(pd_data)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        return fs_sort, fs_acc

# select pca main for numpy
def select_np_pca(np_data, np_label, c=8, g=0.125, cv=5, out_path=now_path, in_path=None):
    if out_path != None:
        if os.path.split(out_path)[1] not in os.listdir(os.path.split(out_path)[0]):
            os.makedirs(out_path)
        # PCA
        pd_data = pd.DataFrame(np_data)
        fs_sort= select_pca(pd_data)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        print('\n特征筛选完成，导出结果中...')
        # plot
        if in_path != None:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-pca.png'), in_path=in_path)
        else:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-pca.png'))
        select_save(os.path.join(out_path, 'Fsort-pca.txt'), fs_sort)
        return fs_acc.index(max(fs_acc))
    else:
        # PCA
        pd_data = pd.DataFrame(np_data)
        fs_sort= select_pca(pd_data)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        return fs_sort, fs_acc


# RF ##########################################################################

# sort
def select_sort_rf(data):
    arr = []
    for i in data:
        arr.append(i)
    index = []
    for i in range(len(arr)):
        index.append(i)
    for i in range(len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        index[min_index], index[i] = index[i], index[min_index]
        arr[min_index], arr[i] = arr[i], arr[min_index]
    # 倒序输出
    re_index = []
    for i in range(len(index) - 1, -1, -1):
        re_index.append(index[i])
    return re_index


# ojld distance
def ojld_distance(p_data, n_data, test_data, number):
    p_distance = []
    n_distance = []
    for line in p_data:
        p_distance.append(math.pow((test_data[number] - line[number]), 2))
    for line in n_data:
        n_distance.append(math.pow((test_data[number] - line[number]), 2))
    return [min(p_distance), min(n_distance)]


# relief method
def select_relief(number, feature_class, feature_line, cycle):
    feature_standard = list(feature_line)
    p_data = []
    n_data = []
    for i in range(len(feature_standard)):
        if feature_class[i] == 0:
            p_data.append(feature_standard[i])
        if feature_class[i] == 1:
            n_data.append(feature_standard[i])
    weight = 0
    m = 0
    for m in range(cycle):
        rand_num = random.randint(0, len(feature_standard) - 1)
        if feature_class[rand_num] == 0:
            distance_box = ojld_distance(p_data, n_data, feature_standard[rand_num], number)
            weight += -distance_box[0] + distance_box[1]
        if feature_class[rand_num] == 1:
            distance_box = ojld_distance(p_data, n_data, feature_standard[rand_num], number)
            weight += -distance_box[1] + distance_box[0]
    aver_weight = weight / (m + 1)
    aver_weight = 1 / (1 + math.exp(-aver_weight))
    return aver_weight


# fscore method
def select_fscore(number, feature_class, feature_line):
    type_both = 0
    type_a = 0
    type_b = 0
    t0 = 0
    t1 = 0
    for i in range(len(feature_class)):
        if feature_class[i] == 0:
            type_a += feature_line[i, number]
            t0 += 1
        else:
            type_b += feature_line[i, number]
            t1 += 1
        type_both += feature_line[i, number]
    avg_0 = type_a / t0
    avg_1 = type_b / t1
    avg_both = type_both / len(feature_class)
    f_son = math.pow(avg_0 - avg_both, 2) + math.pow(avg_1 - avg_both, 2)
    avg_m_0 = 0
    avg_m_1 = 0
    for i in range(len(feature_class)):
        if feature_class[i] == 0:
            avg_m_0 += (math.pow(feature_line[i, number] - avg_0, 2))
        else:
            avg_m_1 += (math.pow(feature_line[i, number] - avg_1, 2))
    f_mother = avg_m_0 / (t0 - 1) + avg_m_1 / (t1 - 1)
    if f_mother != 0:
        f_score = f_son / f_mother
    else:
        f_score = -0.1
    return f_score


# select features
def select_rf(np_data, np_label, cycle):
    relief_list = []
    start_num = 0
    for each_number in range(len(np_data[0])):
        start_num += 1
        ivis.visual_easy_time(start_num, len(np_data[0]))
        type_relief = select_relief(each_number, np_label, np_data, cycle)  # 求得每个特征relief
        type_fscore = select_fscore(each_number, np_label, np_data)  # 求得每个特征f-score
        complex_num = 1 / (math.exp(-type_relief) + 1) + 1 / (math.exp(-type_fscore) + 1)
        relief_list.append(complex_num)
    relief_pool = select_sort_rf(relief_list)  # 排序
    return relief_pool, relief_list


# select rf main for svm
def select_svm_rf(in_path, c=8, g=0.125, cv=5, cycle=50, out_path=now_path, all_p=True, raaBook='minCODE'):
    if out_path != None:
        if os.path.split(out_path)[1] not in os.listdir(os.path.split(out_path)[0]):
            os.makedirs(out_path)
        # load svm
        np_data, np_label = iload.load_svmfile(in_path)
        # rf
        fs_sort, fs_weight = select_rf(np_data, np_label, cycle)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        # plot
        if all_p == True:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-rf.png'), in_path=in_path)
            # plot 2
            iplot.plot_feature_analize(fs_sort, fs_acc, fs_weight, os.path.join(out_path, 'Pie-rf.html'), in_path, raaBook=raaBook)
        else:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-rf.png'))
        select_save(os.path.join(out_path, 'Fsort-rf.txt'), fs_sort)
        return fs_acc.index(max(fs_acc))
    else:
        # load svm
        np_data, np_label = iload.load_svmfile(in_path)
        # rf
        fs_sort, fs_weight = select_rf(np_data, np_label, cycle)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        return fs_sort, fs_acc


# select rf main for numpy
def select_np_rf(np_data, np_label, c=8, g=0.125, cv=5, cycle=50, out_path=now_path, in_path=None, raaBook='minCODE'):
    if out_path != None:
        if os.path.split(out_path)[1] not in os.listdir(os.path.split(out_path)[0]):
            os.makedirs(out_path)
        # rf
        fs_sort, fs_weight = select_rf(np_data, np_label, cycle)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        # plot
        if in_path != None:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-rf.png'), in_path=in_path)
            # plot 2
            iplot.plot_feature_analize(fs_sort, fs_acc, fs_weight, os.path.join(out_path, 'Pie-rf.html'), in_path, raaBook=raaBook)
        else:
            iplot.plot_select(fs_sort, fs_acc, os.path.join(out_path, 'Fsort-rf.png'))
        select_save(os.path.join(out_path, 'Fsort-rf.txt'), fs_sort)
        return fs_acc.index(max(fs_acc))
    else:
        # rf
        fs_sort, fs_weight = select_rf(np_data, np_label, cycle)
        # get filter sort result
        fs_acc = select_test(np_data, np_label, fs_sort, c, g, cv, now_path)
        return fs_sort, fs_acc
