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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最高效实现：
- numba 编译核心计算
- 主进程用 shared_memory 存放当前文件所有 reducefile 数据（按连续块）
- 子进程在 initializer 中 attach 共享内存，接收任务只包含索引和元信息，避免大量序列化
- 批量处理每个任务内多个 raa（batch_size 可调）
- 最大进程数 = max(1, cpu_count - 4)
- tqdm 显示文件与批次进度
注意：你需要在全局提供 iextra.extract_reduce_col_sf, 以及 raacode、pssm_matrixes、reduce 等实际数据。
"""

import os
import math
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from numba import njit, prange

# ---------------------------
# NUMBA 编译的核心计算核
# 请根据你的真实算法把下面函数体改成你原始 kpssm_dt / kpssm_ddt 的逻辑
# 以下为通用示例实现：按列做 k 间隔乘加汇总（与先前示例一致）
# ---------------------------

@njit(parallel=True, fastmath=True)
def kpssm_dt_numba(reducefile_arr, k):
    L, C = reducefile_arr.shape
    out = np.zeros(C, dtype=np.float64)
    max_i = L - k
    for j in prange(C):
        s = 0.0
        for i in range(max_i):
            s += reducefile_arr[i, j] * reducefile_arr[i + k, j]
        out[j] = s
    return out

@njit(parallel=True, fastmath=True)
def kpssm_ddt_numba(reducefile_arr, k, idxs):
    L, C = reducefile_arr.shape
    M = idxs.shape[0]
    out = np.zeros(M, dtype=np.float64)
    max_i = L - k
    for m in prange(M):
        j = idxs[m]
        s = 0.0
        for i in range(max_i):
            s += reducefile_arr[i, j] * reducefile_arr[i + k, j]
        out[m] = s
    return out

# ---------------------------
# 共享内存管理与子进程初始化
# ---------------------------
_shared_shm_name = None
_shared_shape = None
_shared_dtype = None

def _child_init(shm_name, shape, dtype_name):
    global _shared_shm_name, _shared_shape, _shared_dtype, _shared_arr
    _shared_shm_name = shm_name
    _shared_shape = shape
    _shared_dtype = np.dtype(dtype_name)
    shm = shared_memory.SharedMemory(name=_shared_shm_name)
    _shared_arr = np.ndarray(_shared_shape, dtype=_shared_dtype, buffer=shm.buf)
    # _shared_arr 可在子进程中读取；注意不要在子进程关闭共享内存对象（主进程负责 unlink）
    return

def _child_compute_batch(batch_meta):
    """
    batch_meta: list of tuples (start_idx, rows, cols, raa_box_len)
      - start_idx: 起始行在共享大数组中的行索引
      - rows: 每个 reducefile 的行数 (L)
      - cols: 每个 reducefile 的列数 (C)  —— 这里假设同一文件每个 reducefile 的列数相同
      - raa_box_len: 对应 raa_box 长度，用于 ddt 索引（若 batch 包含多个 raa，每个项不同）
    返回：list of feature lists（与 raas 的顺序一致）
    """
    results = []
    k = 3
    # _shared_arr 是主进程写好的二维或三维数组视布局而定
    # 我们在主进程中把每个 reducefile 以相同列数按行块存储：
    # layout: big_arr with shape (N_blocks, L_max, C), but simpler实现为 (N_blocks, L, C) flattened to (N_blocks, L*C)
    # 在这里假设主进程传入 rows, cols 以便正确切片
    for meta in batch_meta:
        start_row, L, C, raa_box_len = meta
        # 取出子数组视图
        flat_segment = _shared_arr[start_row]
        reducefile_arr = flat_segment.reshape((L, C))
        # 调用 numba 内核
        dt = kpssm_dt_numba(reducefile_arr, k)
        ddt = kpssm_ddt_numba(reducefile_arr, k, np.arange(raa_box_len, dtype=np.int32))
        merged = np.concatenate([dt, ddt]).tolist()
        results.append(merged)
    return results

# ---------------------------
# 主函数：feature_kpssm_highest
# ---------------------------

def feature_kpssm(pssm_matrixes, reduce, raacode, batch_size=16):
    """
    参数:
      - pssm_matrixes: iterable of eachfile identifiers
      - reduce: 传给 iextra.extract_reduce_col_sf 的参数
      - raacode: (mapping_dict, iterable_of_raas)
      - batch_size: 每个并行任务包含多少个 raa（较大能减少调度开销）
    返回:
      - kpssm_features: list of per-file mid_matrix (按 raacode[1] 顺序)
    说明:
      - 该实现为每个文件创建共享内存并一次性把所有 reducefile 转入共享内存
      - 子进程只接收索引与元信息进行计算
    """
    cpu_cnt = os.cpu_count() or 1
    workers = max(1, cpu_cnt - 4)

    kpssm_features = []

    for eachfile in tqdm(pssm_matrixes, desc="Files", unit="file", leave=True):
        raas = list(raacode[1])
        N = len(raas)
        if N == 0:
            kpssm_features.append([])
            continue

        # 第一遍主进程读取所有 reducefile，收集它们的 L (rows) 与 C (cols)
        reduce_blocks = []
        rows_list = []
        cols_list = []
        raa_box_lens = []
        for raa in raas:
            raa_box = raacode[0][raa]
            rf = iextra.extract_reduce_col_sf(eachfile, reduce, raa)  # 返回 L x C 的结构
            arr = np.asarray(rf, dtype=np.float64)
            L, C = arr.shape
            reduce_blocks.append(arr)
            rows_list.append(L)
            cols_list.append(C)
            raa_box_lens.append(len(raa_box))

        # 为 simplicity 假设同一文件所有 reducefile 的列数相同（C）
        # 若列数不同，需要把每个 reducefile pad 到同一 C_max（下面代码已支持 pad）
        C_max = max(cols_list)
        L_max = max(rows_list)

        # 构建共享数组： shape (N, L_max*C_max) 使用 dtype float64
        big_shape = (N, L_max * C_max)
        shm = shared_memory.SharedMemory(create=True, size=np.prod(big_shape) * np.dtype(np.float64).itemsize)
        big_arr = np.ndarray(big_shape, dtype=np.float64, buffer=shm.buf)
        # 填充共享数组（按行块）
        for i in range(N):
            arr = reduce_blocks[i]
            L, C = arr.shape
            # 将 arr 放到 (L, C) 区域，剩余用 0 填充
            flat = np.zeros((L_max * C_max,), dtype=np.float64)
            # 把 arr 行主序拷贝到 flat 的前 L*C 位置
            flat[: (L * C)] = arr.reshape(-1)
            big_arr[i, :] = flat

        # 生成每个 reducefile 在共享数组中的元信息 (start_row index is i)
        metas = []
        for i in range(N):
            metas.append( (i, rows_list[i], C_max, raa_box_lens[i]) )

        # 划分批次
        batches = [metas[i:i+batch_size] for i in range(0, N, batch_size)]

        # 为子进程提供共享内存名与 shape/dtype 信息，通过 initializer attach
        shm_name = shm.name
        shm_shape = big_shape
        shm_dtype = np.dtype(np.float64).name

        # 并行计算批次，保证顺序：使用 map 然后展平 batch 结果
        with ProcessPoolExecutor(max_workers=workers, initializer=_child_init,
                                 initargs=(shm_name, shm_shape, shm_dtype)) as ex:
            results_batches = list(tqdm(ex.map(_child_compute_batch, batches), total=len(batches),
                                        desc=f"Computing batches (file)", unit="batch", leave=False))

        # 展平并恢复为按 raas 顺序的 results（每项对应一个 raa）
        results = [item for batch in results_batches for item in batch]
        kpssm_features.append(results)

        # 主进程关闭并 unlink 共享内存
        shm.close()
        shm.unlink()

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
