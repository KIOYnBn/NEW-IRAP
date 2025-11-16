import os
file_path = os.path.dirname(__file__)
import sys
sys.path.append(file_path)
import Extract as iextra
import Visual as ivis
import copy
import itertools

# RAAC-KPSSM ##################################################################
def kpssm_dt(eachfile, k):
    out_box = ivis.visual_create_n_matrix(len(eachfile[0]))
    for i in range(0, len(eachfile) - k):
        now_line = eachfile[i]
        next_line = eachfile[i + k]
        for j in range(len(now_line)):
            out_box[j] += now_line[j] * next_line[j]
    for i in range(len(out_box)):
        out_box[i] = out_box[i] / (len(eachfile) - k - 1)
    return out_box


# kpssm DDT
def kpssm_ddt(eachfile, k, aa_index):
    out_box = []
    for i in range((len(aa_index) - 1) * len(aa_index)):
        out_box.append(0)
    for i in range(0, len(eachfile) - k):
        now_line = eachfile[i]
        next_line = eachfile[i + k]
        n = -1
        for j in range(len(aa_index)):
            next_aa = copy.deepcopy(aa_index)
            now_aa = aa_index[j]
            next_aa.pop(next_aa.index(now_aa))
            for m in next_aa:
                n += 1
                out_box[n] += now_line[now_aa] * next_line[m]
    for i in range(len(out_box)):
        out_box[i] = out_box[i] / (len(eachfile) - k - 1)
    return out_box

# 需要库
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np

# ---------- 1) numba 版本（如果可以修改 kpssm_dt / kpssm_ddt） ----------
# 将计算密集函数用 numba 编译（示例框架，需根据实际实现改写）
from numba import njit, prange

@njit(fastmath=True)
def kpssm_dt_numba(reducefile_arr, k):
    # reducefile_arr: numpy array shape (L, C)
    L, C = reducefile_arr.shape
    # 示例：实际实现替换为你的DT逻辑
    out = np.zeros(C, dtype=np.float64)
    for j in prange(C):
        s = 0.0
        for i in range(L - k):
            s += reducefile_arr[i, j] * reducefile_arr[i + k, j]
        out[j] = s
    return out

@njit(fastmath=True)
def kpssm_ddt_numba(reducefile_arr, k, ddt_index):
    # ddt_index: 1d int array of indices length M
    L, C = reducefile_arr.shape
    M = ddt_index.shape[0]
    out = np.zeros(M, dtype=np.float64)
    for idx in prange(M):
        j = ddt_index[idx]
        s = 0.0
        for i in range(L - k):
            s += reducefile_arr[i, j] * reducefile_arr[i + k, j]
        out[idx] = s
    return out

# ---------- 2) 主进程预处理 + 进程池并行（通用，适用于未能改写为 numba 的情况） ----------
def _worker_compute(args):
    # 入参尽量小：reducefile (numpy array or small list), raa_box_length (int)
    reducefile_arr, raa_box_len = args
    k = 3
    # 如果你有 numba 函数可直接调用；否则调用原 kpssm_dt / kpssm_ddt
    try:
        # 假设 numba 编译的函数已经存在
        dt = kpssm_dt_numba(reducefile_arr, k)
        ddt = kpssm_ddt_numba(reducefile_arr, k, np.arange(raa_box_len, dtype=np.int32))
        # 将结果合并成你原来期望的结构（list/ndarray）
        return np.concatenate([dt, ddt]).tolist()
    except NameError:
        # 回退到原函数（需保证原函数能接受 numpy array）
        dt = kpssm_dt(reducefile_arr, k)
        ddt = kpssm_ddt(reducefile_arr, k, list(range(raa_box_len)))
        return dt + ddt

def feature_kpssm(pssm_matrixes, reduce, raacode, max_workers=None):
    kpssm_features = []
    total_files = len(pssm_matrixes)
    # 遍历文件，主进程一次性准备所有 reducefiles（避免子进程重复 I/O）
    for eachfile in tqdm(pssm_matrixes, desc="Files", unit="file"):
        raas = list(raacode[1])
        args_list = []
        # 主进程做 I/O 并将 reducefile 转为 numpy 数组（更易序列化且供 numba 使用）
        for raa in raas:
            raa_box = raacode[0][raa]
            reducefile = iextra.extract_reduce_col_sf(eachfile, reduce, raa)  # 返回可转换为 numpy 的结构
            reducefile_arr = np.asarray(reducefile, dtype=np.float64)
            args_list.append((reducefile_arr, len(raa_box)))

        # 进程池并行；chunksize 可调，默认取 max(1, len(args)//(workers*4))
        workers = max_workers or None
        chunksize = max(1, len(args_list) // ( (workers or 4) * 4 ))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_worker_compute, args_list, chunksize=chunksize),
                                total=len(args_list), desc=f"Computing per-file", unit="raa", leave=False))
        kpssm_features.append(results)
    return kpssm_features



# RAAC DTPSSM #################################################################
def dtpssm_reduce(raa_box):
    out_box = []
    for i in raa_box:
        out_box.append(i[0])
    return out_box


# top 1 gram
def dtpssm_top_1(reducefile, reduce_index):
    out_box = []
    for line in reducefile:
        out_box.append(reduce_index[line.index(max(line))])
    return out_box


# d 0
def dtpssm_0(fs_top_1, reduce_index):
    out_box = ivis.visual_create_n_matrix(x=len(reduce_index))
    for i in fs_top_1:
        if i in reduce_index:
            out_box[reduce_index.index(i)] += 1
    return out_box


# d n
def dtpssm_n(fs_top_1, reduce_index, d):
    out_box = []
    out_index = []
    for i in reduce_index:
        for j in reduce_index:
            out_box.append(0)
            out_index.append(i + j)
    for i in range(len(fs_top_1) - d):
        if fs_top_1[i] + fs_top_1[i + d] in out_index:
            out_box[out_index.index(fs_top_1[i] + fs_top_1[i + d])] += 1
    return out_box


# PsePSSM主程序
def feature_dtpssm(pssm_matrixes, reduce, raacode):
    dt_features = []
    start_e = 0
    d = 3
    for eachfile in pssm_matrixes:
        start_e += 1
        start_n = 0
        mid_matrix = []
        for raa in raacode[1]:
            start_n += 1
            ivis.visual_detal_time(start_e, len(pssm_matrixes), start_n, len(raacode[1]))
            raa_box = raacode[0][raa]
            reducefile = iextra.extract_reduce_col_sf(eachfile, reduce, raa)
            reduce_index = dtpssm_reduce(raa_box)
            # top_1_gram
            fs_top_1 = dtpssm_top_1(reducefile, reduce_index)
            # d_0
            fs_0 = dtpssm_0(fs_top_1, reduce_index)
            # d_n
            fs_n = []
            for m in range(1, d + 1):
                each_fs_n = dtpssm_n(fs_top_1, reduce_index, m + 1)
                fs_n += each_fs_n
            mid_matrix.append(fs_0 + fs_n)
        dt_features.append(mid_matrix)
    return dt_features


# RAAC SW #####################################################################
def sw_extract(eachfile, eachaaid, lmda):
    eachmatrix = []
    sup_num = int((lmda - 1) / 2)
    sup_matrix = ivis.visual_create_nn_matrix(x=sup_num, y=len(eachfile[0]))
    sup_aaid = []
    for j in range(sup_num):
        sup_aaid.append('X')
    newfile = sup_matrix + eachfile + sup_matrix
    newaaid = sup_aaid + eachaaid + sup_aaid
    for j in range(sup_num, len(newfile) - sup_num):
        select_box = newfile[j - sup_num: j + sup_num + 1]
        eachmatrix.append([newaaid[j]] + select_box)
    return eachmatrix


# Sliding window plus
def sw_plus(eachmatrix):
    aa_index = ivis.visual_create_aa()
    matrix_400 = ivis.visual_create_nn_matrix(x=len(aa_index), y=len(eachmatrix[0][1]))
    for matrix in eachmatrix:
        if matrix[0] in aa_index:
            for line in matrix[1:]:
                for m in range(len(line)):
                    matrix_400[aa_index.index(matrix[0])][m] += line[m]
    for j in range(len(matrix_400)):
        for k in range(len(matrix_400[j])):
            matrix_400[j][k] = float('%.3f' % matrix_400[j][k])
    return matrix_400


# Sliding window reduce
def sw_reduce(raacode, eachplus, reduce):
    aa_index = ivis.visual_create_aa()
    reduce_fs = []
    for raa in raacode[1]:
        mid_box = iextra.extract_reduce_row_sf(eachplus, aa_index, reduce, raa)
        mid_box = iextra.extract_reduce_col_sf(mid_box, reduce, raa)
        out_box = []
        for i in mid_box:
            out_box += i
        reduce_fs.append(out_box)
    return reduce_fs


# Sliding window main
def feature_sw(pssm_matrixes, pssm_aaid, reduce, raacode):
    sw_features = []
    lmda = '5'
    start_e = 0
    for i in range(len(pssm_matrixes)):
        start_e += 1
        ivis.visual_easy_time(start_e, len(pssm_matrixes))
        eachfile = pssm_matrixes[i]
        eachaaid = pssm_aaid[i]
        # 提取 L - lmda 个窗口
        eachmatrix = sw_extract(eachfile, eachaaid, int(lmda))
        # 合并20种aa为20x20维矩阵
        eachplus = sw_plus(eachmatrix)
        # 约化矩阵
        reduce_fs = sw_reduce(raacode, eachplus, reduce)
        sw_features.append(reduce_fs)
    return sw_features


# RAAC KMER ################################################################### 
def create_kmer_dict(simple_raa, kmer):
    kmer_dcit = {}
    for i in itertools.product("".join(simple_raa), repeat=kmer):
        j_str = ""
        for j in i:
            j_str = j_str + j
        kmer_dcit[j_str] = 0
    return kmer_dcit


# reduce
def kmer_reduce(raa_box, eachfile):
    out_box = []
    for i in eachfile:
        for j in raa_box:
            if i in j:
                out_box.append(j[0])
    simple_raa = []
    for i in raa_box:
        simple_raa.append(i[0])
    return out_box, simple_raa


def psekraac_reduce(raacode, raa, line):
    raa_box = raacode[0][raa]
    reduce_box = kmer_reduce(raa_box, line)
    reducefile = reduce_box[0]
    simple_raa = reduce_box[1]
    return reducefile, simple_raa


def kmer_count(simple_raa, line, kmer, gap, r):
    line = "".join(line)
    kmer_dcit = create_kmer_dict(simple_raa, kmer)
    kmer_dcit_c = kmer_dcit.copy()
    for i in range(0, len(line) - kmer*(r+1) + r + 1, gap + 1):
        kmer_dcit_c[line[i:i+kmer*(r+1):r+1]] = kmer_dcit_c[line[i:i+kmer*(r+1):r+1]] + 1
    return kmer_dcit_c


def kmer_change(data):
    out = []
    for key in data:
        out.append(data[key])
    return out


def kmer_part(raacode, pssm_aaid, k, g, lmda):
    sq_box = {}
    for i in range(len(pssm_aaid)):
        line = "".join(pssm_aaid[i])
        sq_box[i] = line
    out_box = []
    for line in pssm_aaid:
        mid_raa = []
        for raa in raacode[1]:
            reducefile, simple_raa = psekraac_reduce(raacode, raa, line)
            k_fs = kmer_count(simple_raa, reducefile, k, g, lmda)
            k_fs = kmer_change(k_fs)
            mid_raa.append(k_fs)
        out_box.append(mid_raa)
    return out_box


def psekraac_sum(k_fs, kg_fs, kgl_fs):
    out_box = []
    for i in range(len(k_fs)):
        raa_box_1 = k_fs[i]
        raa_box_2 = kg_fs[i]
        raa_box_3 = kgl_fs[i]
        new_box = []
        for j in range(len(raa_box_1)):
            dic_raa_1 = raa_box_1[j]
            dic_raa_2 = raa_box_2[j]
            dic_raa_3 = raa_box_3[j]
            new_dic = []
            for k in dic_raa_1:
                new_dic.append(k)
            for k in dic_raa_2:
                new_dic.append(k)
            for k in dic_raa_3:
                new_dic.append(k)
            new_box.append(new_dic)
        out_box.append(new_box)
    return out_box


# psekraac main
def feature_kmer(pssm_aaid, raacode):
    k = 2
    g = 0
    lmda = 1
    # k
    k_fs = kmer_part(raacode, pssm_aaid, k, 0, 1)
    # k,g
    kg_fs = kmer_part(raacode, pssm_aaid, k, g, 1)
    # k,g,lmda
    kgl_fs = kmer_part(raacode, pssm_aaid, k, g, lmda)
    # sum
    psekraac_features = psekraac_sum(k_fs, kg_fs, kgl_fs)
    return psekraac_features


# AAC #########################################################################
def feature_oaac(pssm_aaid, raacode):
    all_features = []
    start_e = 0
    for line in pssm_aaid:
        start_e += 1
        ivis.visual_easy_time(start_e, len(pssm_aaid))
        mid_box = []
        for raa in raacode[1]:
            raa_box = raacode[0][raa]
            aabox = ivis.visual_create_n_matrix(len(raa_box))
            for i in line:
                for j in raa_box:
                    if i in j:
                        aabox[raa_box.index(j)] += 1
            mid_box.append(aabox)
        all_features.append(mid_box)
    return all_features


# SAAC ########################################################################
def saac_count(file_box, simple_raa):
    out = []
    for i in file_box:
        aabox = ivis.visual_create_n_matrix(len(simple_raa))
        for j in i:
            for k in simple_raa:
                if j == k:
                    aabox[simple_raa.index(k)] += 1
        out.append(aabox)
    return out


def saac_el(eachfile, simple_raa):
    dn = 25
    dc = 10
    dn_1 = eachfile[0:4 * dn]
    dc_1 = eachfile[-dc:]
    dl_1 = eachfile[4 * dn:-dc]
    all_elfs = saac_count([dn_1, dl_1, dc_1], simple_raa)
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# SAAC extract middle sequence
def saac_em(eachfile, simple_raa):
    dn = 25
    dc = 10
    dn_1 = eachfile[0:4 * dn]
    dc_1 = eachfile[-dc:]
    dl_1 = eachfile[-(dc + 20):-dc]
    all_elfs = saac_count([dn_1, dl_1, dc_1], simple_raa)
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# SAAC extract short sequence
def saac_es(eachfile, simple_raa):
    dc = 10
    dn = int((len(eachfile) - dc) / 2)
    dn_1 = eachfile[0:dn]
    dc_1 = eachfile[-dc:]
    dl_1 = eachfile[dn:-dc]
    all_elfs = saac_count([dn_1, dc_1, dl_1], simple_raa)
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# saac mian
def feature_saac(pssm_aaid, raacode):
    dn = 25
    dc = 10
    start_e = 0
    all_features = []
    for eachfile in pssm_aaid:
        start_e += 1
        start_n = 0
        mid_box = []
        for raa in raacode[1]:
            start_n += 1
            ivis.visual_detal_time(start_e, len(pssm_aaid), start_n, len(raacode[1]))
            raa_box = raacode[0][raa]
            out_file, simple_raa = kmer_reduce(raa_box, eachfile)
            each_fs = []
            if len(out_file) >= (4 * dn + dc + 20):
                each_fs = saac_el(out_file, simple_raa)
            if (4 * dn + dc) < len(out_file) < (4 * dn + dc + 20):
                each_fs = saac_em(out_file, simple_raa)
            if len(out_file) <= (4 * dn + dc):
                each_fs = saac_es(out_file, simple_raa)
            mid_box.append(each_fs)
        all_features.append(mid_box)
    return all_features
