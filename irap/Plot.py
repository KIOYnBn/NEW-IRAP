# import packages
import os
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Pie, Bar, Sankey
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
from itertools import chain
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
import Load as iload
import Reduce as ired
import Feature as ifeat
import Visual as ivis
now_path = os.getcwd()
raac_path = os.path.join(file_path, 'raacDB')


# feature filter ##############################################################
# 建立索引
def plot_select_id(len_list, method_fs):
    out = {}
    t = -1
    for i in range(len(method_fs)):
        for j in range(len_list[method_fs[i]]):
            t += 1
            out[t] = method_fs[i]
    return out


# 特征分类
def plot_select_class(standard, size_fs, method_fs, k):
    len_list = [int(math.pow(size_fs, 2)), int(math.pow(size_fs, 2)), int(3*math.pow(size_fs, 2)+size_fs), int(math.pow(size_fs, 2)), int(3*math.pow(size_fs, 2)), 3*size_fs, size_fs]
    fs_dict = plot_select_id(len_list, method_fs)
    pssm_value = []
    kpssm_value = []
    dtpssm_value = []
    sw_value = []
    kmer_value = []
    saac_value = []
    aac_value = []
    pmv = 0
    kmv = 0
    dmv = 0
    swv = 0
    krv = 0
    scv = 0
    acv = 0
    for i in standard:
        if fs_dict[i] == 0:
            pmv += 1
        if fs_dict[i] == 1:
            kmv += 1
        if fs_dict[i] == 2:
            dmv += 1
        if fs_dict[i] == 3:
            swv += 1
        if fs_dict[i] == 4:
            krv += 1
        if fs_dict[i] == 5:
            scv += 1
        if fs_dict[i] == 6:
            acv += 1
        pssm_value.append(pmv / (1.5 * len(standard)))
        kpssm_value.append(kmv / (1.5 * len(standard)))
        dtpssm_value.append(dmv / (1.5 * len(standard)))
        sw_value.append(swv / (1.5 * len(standard)))
        kmer_value.append(krv / (1.5 * len(standard)))
        saac_value.append(scv / (1.5 * len(standard)))
        aac_value.append(acv / (1.5 * len(standard)))
    return pssm_value, kpssm_value, dtpssm_value, sw_value, kmer_value, saac_value, aac_value


# 绘制折线图
def plot_select_visual(data, all_values, type_p):
    x = []
    y = []
    for i in range(len(data)):
        x.append(i + 1)
    for j in data:
        y.append(j)
    plt.figure()
    plt.plot(x, y, label='ACC')
    plt.plot(x, all_values[0], color='blue', label='PSSM')
    plt.plot(x, all_values[1], color='green', label='KPSSM')
    plt.plot(x, all_values[2], color='yellow', label='DTPSSM')
    plt.plot(x, all_values[3], color='red', label='SW')
    plt.plot(x, all_values[4], color='pink', label='KMER')
    plt.plot(x, all_values[5], color='orange', label='SAAC')
    plt.plot(x, all_values[6], color='gray', label='OAAC')
    plt.legend(bbox_to_anchor=(0., 1.03, 1., .102), loc=0, ncol=4, mode="expand", borderaxespad=0.)
    plt.xlabel("Feature Number")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_x = y.index(max(y))
    max_y = max(y)
    max_pmv = all_values[0][max_x]
    max_kmv = all_values[1][max_x]
    max_dmv = all_values[2][max_x]
    max_swv = all_values[3][max_x]
    max_krv = all_values[4][max_x]
    max_scv = all_values[5][max_x]
    max_acv = all_values[6][max_x]
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)
    plt.text(max_x, max_pmv, str(int(max_pmv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_kmv, str(int(max_kmv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_dmv, str(int(max_dmv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_swv, str(int(max_swv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_krv, str(int(max_krv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_scv, str(int(max_scv * 1.5 * len(data))), fontsize=6)
    plt.text(max_x, max_acv, str(int(max_acv * 1.5 * len(data))), fontsize=6)


def plot_select_simple(data, type_p):
    x = []
    y = []
    for i in range(len(data)):
        x.append(i + 1)
    for j in data:
        y.append(j)
    plt.figure()
    plt.plot(x, y, label='ACC')
    plt.xlabel("Feature Number")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_x = y.index(max(y))
    max_y = max(y)
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)


# filter main
def plot_select(relief_pool, fs_acc, out_path, in_path=None, k=2):
    if in_path != None:
        size_fs = int(os.path.split(in_path)[-1].split('-')[1].split('s')[-1])
        method_fs = [ int(x) for x in list(os.path.split(in_path)[-1].split('-')[-1].split('.')[0]) ]
        # ifs
        all_values = plot_select_class(relief_pool, size_fs, method_fs, k)
        plot_select_visual(fs_acc, all_values, 'IFS-Acc\n\n\n')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        # ifs
        plot_select_simple(fs_acc, 'IFS-Acc')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')


# feature analize #############################################################

# fs id
def plot_fa_id(raacode, method_fs, fs_sort, fs_acc, fs_weight, k):
    # raaPSSM
    raapssm = []
    for i in raacode:
        for j in raacode:
            raapssm.append(i[0] + j[0])
    raapssm_id = ivis.visual_create_n_matrix(x=len(raapssm), fill=0)
    # AAC
    aac = []
    for i in raacode:
        aac.append(i[0])
    aac_id = ivis.visual_create_n_matrix(x=len(aac), fill=6)
    # SAAC
    saac = []
    for i in range(3):
        for j in raacode:
            saac.append(j[0])
    saac_id = ivis.visual_create_n_matrix(x=len(saac), fill=5)
    # raaKmer
    mid = list(ifeat.create_kmer_dict(aac, k))
    raakmer = mid + mid + mid
    raakmer_id = ivis.visual_create_n_matrix(x=len(raakmer), fill=4)
    # raaKPSSM
    raakpssm = raapssm
    raakpssm_id = ivis.visual_create_n_matrix(x=len(raakpssm), fill=1)
    # raaSW
    raasw = raapssm
    raasw_id = ivis.visual_create_n_matrix(x=len(raasw), fill=3)
    # raaDTPSSM
    raadtpssm = aac + raakmer
    raadtpssm_id = ivis.visual_create_n_matrix(x=len(raadtpssm), fill=2)
    all_raa = [raapssm, raakpssm, raadtpssm, raasw, raakmer, saac, aac]
    all_id = [raapssm_id, raakpssm_id, raadtpssm_id, raasw_id, raakmer_id, saac_id, aac_id]
    out_box = []
    out_id = []
    for i in method_fs:
        out_box = out_box + all_raa[i]
        out_id = out_id + all_id[i]
    # dict
    out_dict = []
    for i in range(len(fs_sort)):
        out_dict.append([out_box[fs_sort[i]], fs_sort[i], fs_weight[fs_sort[i]], fs_acc[i], out_id[fs_sort[i]]])
    return out_dict


# 饼图
def plot_fa_pie(data, out):
    c = (
        Pie()
        .add(
            "",
            data,
            radius=["30%", "75%"],
            center=["50%", "50%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="The first 200-dimensional feature distribution",
                                                   pos_left='bottom'), toolbox_opts=opts.ToolboxOpts(
                is_show=True, pos_top="top", pos_left="right", feature={"saveAsImage": {}})
        )
        .render(out)
    )
    return c


def plot_fa_visual(feature_id, method_fs, out):
    data = [['PSSM', 0], ['KPSSM', 0], ['DTPSSM', 0], ['SW', 0], ['KMER', 0], ['SAAC', 0], ['OAAC', 0]]
    for i in range(int(len(feature_id)*0.1)):
        data[feature_id[i][-1]][-1] += 1
    plot_fa_pie(data, out)


def plot_fa_out(out, feature_id):
    out_line = 'Index\tFeature ID\tFeature weight\t5Flod-ACC'
    for i in range(len(feature_id)):
        out_line += '\n' + str(i+1) + '\t' + feature_id[i][0] + '\t' + str(feature_id[i][2]) + '\t' + str(feature_id[i][3])
    with open(out, 'w') as f:
        f.write(out_line)


# feature analize main
def plot_feature_analize(fs_sort, fs_acc, fs_weight, out_path, in_path, raaBook='raaCODE', k=2):
    raa_dict = iload.load_raac(os.path.join(raac_path, raaBook))
    raacode = raa_dict[0][os.path.split(in_path)[-1].split('-')[0]+'-'+os.path.split(in_path)[-1].split('-')[1]]
    method_fs = [ int(x) for x in list(os.path.split(in_path)[-1].split('-')[-1].split('.')[0]) ]
    # feature id
    feature_id = plot_fa_id(raacode, method_fs, fs_sort, fs_acc, fs_weight, k)
    # 特征可视化
    plot_fa_visual(feature_id, method_fs, out_path)
    # 特征列表输出
    plot_fa_out(os.path.join(os.path.split(out_path)[0], 'Feature_Weight.xls'), feature_id)
    
    
# evaluate plot ###############################################################
# sort
def plot_eval_sort(box_v, box_i, type_b):
    out_v = []
    out_i = []
    for i in range(len(box_i)):
        out_v.append(box_v[i])
        out_i.append(int(box_i[i].strip(type_b)))
    n = len(out_i)
    for i in range(n):
        for j in range(0, n-i-1):
            if out_i[j] > out_i[j+1]:
                out_i[j], out_i[j+1] = out_i[j+1], out_i[j]
                out_v[j], out_v[j+1] = out_v[j+1], out_v[j]
    for i in range(len(out_i)):
        out_i[i] = type_b + str(out_i[i])
    return out_v, out_i


# 柱状图
def plot_eval_histogram(data, d_index, d_class, out, type_r):
    new_index = []
    for i in range(len(d_index)):
        new_index.append(d_index[i].strip(type_r))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 汉显
    plt.rcParams['axes.unicode_minus'] = False  # 汉显
    plt.xlabel(d_class, fontsize=10)  # X轴标题
    plt.ylabel('ACC(%)', fontsize=10)  # Y轴标题
    plt.figure(figsize=(int(len(d_index)*0.2)+3, 5))
    plt.bar(new_index, data, width=0.8)  # 数据
    plt.title('ACC of each ' + d_class)  # 标题
    plt.grid(axis="y", c='g', linestyle='dotted')
    plt.savefig(os.path.join(out, 'ACC_' + d_class + '.png'), dpi=300)  # 保存
    plt.close()


# 密度图
def plot_eval_density(eval_value, out):
    values = []
    for key in eval_value:
        values.append(float(key[4]) * 100)
    s = pd.Series(values)
    sns.kdeplot(s)
    plt.title('ACC Density')
    plt.savefig(os.path.join(out, 'ACC_Density.png'), dpi=300)
    plt.close()


# 热力图
def plot_eval_heatmap(eval_value, eval_key, out, t_s_index, s_s_index):
    data = {}
    for i in range(len(eval_value)):
        data[eval_key[i]] = float('%.4f' % (eval_value[i][4]))
    map_box = []
    min_num = float(data['t0s20']) * 100
    for key in data:
        if float(data[key]) * 100 > min_num:
            pass
        else:
            min_num = float(data[key]) * 100
    for s in s_s_index:
        mid_box = []
        for t in t_s_index:
            if t + s in data:
                mid_box.append(float('%.4f' % data[t + s]) * 100)
            else:
                mid_box.append(min_num - 10)
        map_box.append(mid_box)
    f, ax = plt.subplots(figsize=(int(len(t_s_index)*0.3)+4, 10))
    x = np.array(map_box)
    ax.set_title('ACC_Heatmap')
    ax.set_ylabel('Size')
    ax.set_xlabel('Type')
    sns.heatmap(x, cmap='YlGnBu', annot=True, mask=(x < min_num), vmax=100, linewidths=0.1, square=False,
                xticklabels=True, yticklabels=True)
    ax.set_xticklabels(t_s_index)
    ax.set_yticklabels(s_s_index)
    plt.savefig(os.path.join(out, 'ACC_Heatmap.png'), dpi=400)
    plt.close()


# main
def plot_evaluate(cluster_t, t_index, out, cluster_s, s_index, eval_value, eval_key):
    # histogram
    cluster_s_t, t_s_index = plot_eval_sort(cluster_t, t_index, 't')
    plot_eval_histogram(cluster_s_t, t_s_index, 'Type', out, 't')
    cluster_s_s, s_s_index = plot_eval_sort(cluster_s, s_index, 's')
    plot_eval_histogram(cluster_s_s, s_s_index, 'Size', out, 's')
    # density
    plot_eval_density(eval_value, out)
    # heatmap
    plot_eval_heatmap(eval_value, eval_key, out, t_s_index, s_s_index)
    

# ROC #########################################################################
def plot_roc_svm(af_data, af_label, c_number, ga):
    # 分割数据
    train_data, test_data, train_label, test_label = model_selection.train_test_split(af_data, af_label, test_size=.3,
                                                                                      random_state=0)
    # svm分类训练
    roc = svm.SVC(kernel='rbf', C=c_number, gamma=ga, probability=True)
    test_predict_label = roc.fit(train_data, train_label).decision_function(test_data)
    # roc坐标获取
    fpr, tpr, threshold = roc_curve(test_label, test_predict_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_line(fpr, tpr, roc_auc, out):
    plt.figure()
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(out, dpi=300)


# ROC main
def plot_roc(file, out=now_path, c=8, g=0.125):
    np_data, np_label = iload.load_svmfile(file)
    fpr, tpr, roc_auc = plot_roc_svm(np_data, np_label, c, g)
    if out != now_path:
        plot_roc_line(fpr, tpr, roc_auc, out)
    else:
        plot_roc_line(fpr, tpr, roc_auc, os.path.join(out, 'ROC-cruve.png'))


# weblogo #####################################################################
def plot_weblogo_check(line):
    add = 0
    for i in line:
        add += i
    if add > 100:
        mid = add - 100
        line[line.index(max(line))] = line[line.index(max(line))] - mid
    elif add < 100:
        mid = 100 - add
        line[line.index(max(line))] = line[line.index(max(line))] + mid
    return line


# weblogo change
def plot_weblogo_change(matrix, raacode):
    # 合并列
    new_matrix = []
    aa = 'ARNDCQEGHILKMFPSTWYV'
    for line in matrix:
        new_line = ivis.visual_create_n_matrix(x=len(raacode), fill=0)
        for i in range(len(line)):
            for j in range(len(raacode)):
                if aa[i] in raacode[j]:
                    new_line[j] += line[i]
        new_line = plot_weblogo_check(new_line)
        new_matrix.append(new_line)
    # 提取列
    out_box = []
    for i in range(len(new_matrix[0])):
        mid_box = []
        for j in new_matrix:
            mid_box.append(j[i])
        out_box.append(mid_box)
    # 格式转换
    out_dic = []
    for each in out_box:
        mid_box = []
        for i in each:
            mid_dic = {"value": i, "percent": i / 100}
            mid_box.append(mid_dic)
        out_dic.append(mid_box)
    return out_dic


def plot_weblogo_draw(site_list, type_list, type_value, out):
    c = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
    c.add_xaxis(site_list)
    for i in range(len(type_list)):
        c.add_yaxis(type_list[i], type_value[i], stack="stack1", category_gap="50%")
    c.set_series_opts(
        label_opts=opts.LabelOpts(
            is_show=False,
            position="right",
            formatter=JsCode(
                "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
            )
        )
    if len(type_list) >= 8:
        c.set_global_opts(
            title_opts=opts.TitleOpts(title="Sequence Reduce Weblogo", pos_left="center"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            yaxis_opts=opts.AxisOpts(name="persernt(%)"),
            xaxis_opts=opts.AxisOpts(name="site"),
            toolbox_opts=opts.ToolboxOpts(is_show=True, pos_left="0px", pos_bottom="0px", feature={"saveAsImage": {}}),
            legend_opts=opts.LegendOpts(pos_right='0px', pos_top='10px', orient='vertical')
            )
    else:
        c.set_global_opts(
            title_opts=opts.TitleOpts(title="Sequence Reduce Weblogo", pos_left="center"),
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
            yaxis_opts=opts.AxisOpts(name="persernt(%)"),
            xaxis_opts=opts.AxisOpts(name="site"),
            toolbox_opts=opts.ToolboxOpts(is_show=True, pos_left="0px", pos_bottom="0px", feature={"saveAsImage": {}}),
            legend_opts=opts.LegendOpts(pos_top='5%')
            )
    c.render(out)
    return c


# weblogo main
def plot_weblogo(file, raa, reduce, out):
    raa_file = os.path.join(raac_path, raa)
    raa_dict, raa_index = iload.load_raac(raa_file)
    raacode = raa_dict[reduce]
    matrix, sq = iload.load_pssm(file)
    matrix = iload.load_weblogo(matrix)
    type_value = plot_weblogo_change(matrix, raacode)
    site_list = []
    for i in range(len(type_value[0])):
        site_list.append(i+1)
    plot_weblogo_draw(site_list, raacode, type_value, out)


# SSC #########################################################################
# ssc cluster
def plot_ssc_cluster(source, target):
    sl, tl, vl = [], [], []
    for ti, taac in enumerate(target):
        taa_set = set(taac)
        aac_len = len(taac)
        for si, saac in enumerate(source):
            intersect = taa_set & set(saac)
            if intersect:
                sl.append(si)
                tl.append(ti)
                vl.append(len(intersect))
                aac_len -= len(intersect)
            if aac_len == 0:
                break
    return sl, tl, vl


# link
def plot_ssc_link(clusters):
    base_idx = 0
    source_idx, target_idx, values = [], [], []
    for i in range(len(clusters)-1):
        sl, tl, vl = plot_ssc_cluster(clusters[i], clusters[i+1])
        sidx = [i+base_idx for i in sl]
        base_idx += len(clusters[i])
        tidx = [i+base_idx for i in tl]
        source_idx.extend(sidx)
        target_idx.extend(tidx)
        values.extend(vl)
    return source_idx, target_idx, values


# sourse
def plot_ssc_sourse(labels, source_idx, target_idx, values):
    linkes = []
    for i in range(len(source_idx)):
        x_1 = source_idx[i]
        x_2 = target_idx[i]
        x_3 = values[i]
        mid_dic = {"source": labels[x_1], "target": labels[x_2], "value": x_3}
        if labels[x_1] != labels[x_2] and len(labels[x_1]) < len(labels[x_2]):
            linkes.append(mid_dic)
    return linkes


# nodes
def plot_ssc_nodes(linkes):
    name_box = []
    for dic in linkes:
        if dic["source"] not in name_box:
            name_box.append(dic["source"])
        if dic["target"] not in name_box:
            name_box.append(dic["target"])
    nodes = []
    for i in name_box:
        mid_dic = {"name": i}
        nodes.append(mid_dic)
    return nodes


# 绘制sankey图
def plot_ssc_sankey(nodes, linkes, title_ssc, out):
    c = (
        Sankey()
        .add(
            title_ssc,
            nodes,
            linkes,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="ReduceMap"), toolbox_opts=opts.ToolboxOpts(
                is_show=True, pos_top="top", pos_left="right", feature={"saveAsImage": {}})
            )
        .render(out)
    )
    print("ReduceMap has been saved at " + c)


# ssc main
def plot_ssc(file, type_r, out=now_path):
    raa_file = os.path.join(raac_path, file)
    raac_list = iload.load_ssc(raa_file, type_r)
    # get linkes
    source_idx, target_idx, values = plot_ssc_link(raac_list)
    labels = list(chain(*raac_list))
    linkes = plot_ssc_sourse(labels, source_idx, target_idx, values)
    # get nodes
    nodes = plot_ssc_nodes(linkes)
    # plot SSC
    title_ssc = "type" + type_r
    out_path = os.path.join(out, file + "_type" + type_r + "_SSC.html")
    plot_ssc_sankey(nodes, linkes, title_ssc, out_path)
