# import packages
import os
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import Load as iload
import Visual as ivis
import SVM as isvm
import Plot as iplot
from math import sqrt


# get scores
def evaluate_score(true, tfpn):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(true)):
        if tfpn[j] == 0.0 and true[j] == 0.0:
            tp += 1
        if tfpn[j] == 1.0 and true[j] == 1.0:
            tn += 1
        if tfpn[j] == 0.0 and true[j] == 1.0:
            fp += 1
        if tfpn[j] == 1.0 and true[j] == 0.0:
            fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    if (tp + fn) == 0 or (tn + fp) == 0 or (tp + fp) == 0 or (tn + fn) == 0:
        return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % 0), float('%.3f' % 0), float('%.3f' % 0), float(
            '%.3f' % 0)
    else:
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        ppv = tp / (tp + fp)
        mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
        return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % sn), float('%.3f' % sp), float(
            '%.3f' % ppv), float('%.3f' % mcc)


# cluster
def evaluate_cluster(eval_value, eval_key):
    t_index = []
    s_index = []
    acc_score = {}
    for i in range(len(eval_value)):
        acc_score[eval_key[i]] = float('%.3f' % eval_value[i][4])
    for key in eval_key:
        t_tp = key.split('s')[0]
        s_tp = 's' + key.split('s')[1]
        if t_tp not in t_index:
            t_index.append(t_tp)
        if s_tp not in s_index:
            s_index.append(s_tp)
    cluster_t = ivis.visual_create_n_matrix(x=len(t_index))
    cluster_s = ivis.visual_create_n_matrix(x=len(s_index))
    t_number = ivis.visual_create_n_matrix(x=len(t_index))
    s_number = ivis.visual_create_n_matrix(x=len(s_index))
    for key in acc_score:
        t_tp = key.split('s')[0]
        s_tp = 's' + key.split('s')[1]
        cluster_t[t_index.index(t_tp)] += acc_score[key]
        cluster_s[s_index.index(s_tp)] += acc_score[key]
        t_number[t_index.index(t_tp)] += 1
        s_number[s_index.index(s_tp)] += 1
    for i in range(len(cluster_t)):
        cluster_t[i] = float('%.4f' % (cluster_t[i] / t_number[i]))
    for i in range(len(cluster_s)):
        cluster_s[i] = float('%.4f' % (cluster_s[i] / s_number[i]))
    return cluster_t, cluster_s, t_index, s_index


# evaluate main
def evaluate_file(file, c=8, g=0.125, cv=5, out=None):
    test_label, predict_label = isvm.svm_evaluate(file, c, g, cv)
    score_line = evaluate_score(test_label, predict_label)
    if out != None:
        line_num = ''
        for li in range(len(score_line)):
            each_num = score_line[li]
            if li < 4:
                line_num += str(int(each_num)) + '\t'
            else:
                line_num += str('%.3f' % each_num) + '\t'
        lines = 'Model Evaluation\n\ntp\ttn\tfp\tfn\tacc\tsn\tsp\tppv\tmcc\n' + line_num[:-1]
        lines += '\n\n' + ivis.visual_eval_analize()
        with open(out, 'w', encoding='utf-8') as f1:
            f1.write(lines)
            f1.close()
    else:
        return [('tp', 'tn', 'fp', 'fn', 'acc', 'sn', 'sp', 'ppv', 'mcc'), score_line]


# evaluate main for folder
def evaluate_folder(path, cg=None, cv=5, out=None):
    eval_value = []
    eval_key = []
    best_rap = ''
    if cg != None:
        cg_box = iload.load_hys(cg)
        for i in path:
            best_acc = 0
            if cg_box[i][1] == 0:
                cg_box[i][1] = 0.01
            test_label, predict_label = isvm.svm_evaluate(i, cg_box[i][0], cg_box[i][1], cv)
            eval_value.append(evaluate_score(test_label, predict_label))
            eval_key.append(os.path.split(i)[-1].split('-')[1])
            if eval_value[-1][4] >= best_acc:
                best_rap = i
                best_acc = eval_value[-1][4]
    else:
        for i in path:
            best_acc = 0
            test_label, predict_label = isvm.svm_evaluate(i, 8, 0.125, cv)
            eval_value.append(evaluate_score(test_label, predict_label))
            eval_key.append(os.path.split(i)[-1].split('-')[1])
            if eval_value[-1][4] >= best_acc:
                best_rap = i
                best_acc = eval_value[-1][4]
    if out != None:
        if os.path.split(out)[-1] not in os.listdir(os.path.split(out)[0]):
            os.makedirs(out)
        # save file
        out_lines = ''
        for i in range(len(eval_value)):
            line = eval_value[i]
            mid_line = eval_key[i]
            for j in line:
                mid_line += ',' + str(j)
            out_lines += mid_line + '\n'
        out_lines = 'index,tp,tn,fp,fn,acc,sn,sp,ppv,mcc\n' + out_lines
        with open(os.path.join(out, 'Features_eval.csv'), 'w') as f2:
            f2.write(out_lines)
        # plot
        cluster_t, cluster_s, t_index, s_index = evaluate_cluster(eval_value, eval_key)
        iplot.plot_evaluate(cluster_t, t_index, out, cluster_s, s_index, eval_value, eval_key)
        return os.path.split(best_rap)[-1]
    else:
        return [('tp', 'tn', 'fp', 'fn', 'acc', 'sn', 'sp', 'ppv', 'mcc')] + eval_value
