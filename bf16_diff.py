import os
import sys
import math
from tabnanny import verbose
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('error')

def bfloat16_to_float32(bf16_array):
    """将 numpy 数组中的 bfloat16 转换为 float32"""
    # 将 bf16 拆分为 uint16 进行位操作
    bf16_as_uint16 = bf16_array.view(np.uint16)
    
    # 创建一个新的 float32 数组
    fp32_array = np.zeros_like(bf16_as_uint16, dtype=np.float32)
    
    # 将 bf16 的高 16 位扩展到 float32 的高位
    fp32_as_uint32 = bf16_as_uint16.astype(np.uint32) << 16
    
    # 将这些位视为 float32 的位模式
    fp32_array.view(np.uint32)[:] = fp32_as_uint32
    
    return fp32_array

def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return dist

def color_print(state, str):
    color_char = '32' if state == True else '31'
    color_prefix = '\033[1;%sm' % (color_char)
    color_postfix = '\033[0m'
    print(color_prefix + str + color_postfix)

def fp_diff(a, b, abs_thresh = 1e-6, rel_thresh=1e-6, verbose_cnt = 0):
    aa = a
    bb = b
    print(aa.shape)
    print(bb.shape)
    
    
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    if a.shape[0] != b.shape[0]:
        color_print(False, 'diff failed: data len not equal, %d vs %d' % (a.shape[0], b.shape[0]))
        return False
    len = a.shape[0]

    if len <= 0:
        return True
    
    abs_diff = np.abs(a - b)
    # print(np.maximum(np.abs(a), np.abs(b)))
    try:
        rel_diff = abs_diff / np.maximum(np.abs(a), np.abs(b))
    except Warning as e:
        rel_diff = -9999

    global max_abs_diff
    global max_rel_diff
    global avg_abs_diff
    global avg_rel_diff

    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    avg_abs_diff = np.average(abs_diff)
    avg_rel_diff = np.average(rel_diff)
    cos_simulate = 1 - cosine_distance(a, b)
   # cos_simulate = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

    print('max_abs_diff: %f' % (max_abs_diff))
    print('max_rel_diff: %f' % (max_rel_diff))
    print('avg_abs_diff: %f' % (avg_abs_diff))
    print('avg_rel_diff: %f' % (avg_rel_diff))
    print('cos_simulate: ' , (cos_simulate))


    # diff_point = ((abs_diff > abs_thresh) & (rel_diff > rel_thresh)) | np.isnan(a) | np.isnan(b)
    diff_point = (abs_diff > abs_thresh) | np.isnan(a) | np.isnan(b)
    diff_num = np.sum(diff_point)

    if diff_num > 0:
        color_print(False, 'diff failed: max abs diff is %f (thresh = %f), max rel diff is %f (thresh = %f), diff_num = %d' % (max_abs_diff, abs_thresh, max_rel_diff, rel_thresh, diff_num))
        verbose_err_num = min(diff_num, verbose_cnt)
        if verbose_err_num > 0:
            err_cnt = 0
            for i in range(len):
                if diff_point[i] == True:
                    if err_cnt <= min(diff_num, verbose_cnt):
                        color_print(False, 'diff at %d: %f vs %f, %f' % (i, a[i], b[i], a[i] - b[i]))
                    err_cnt += 1
                else:
                    #color_print(True, '                                  same at %d: %f vs %f, %f' % (i, a[i], b[i], a[i] - b[i]))
                    pass
                    
            color_print(False, 'total %d data diff' % (diff_num))
        else:
            color_print(False, 'diff verbose has been disabled, set option --diff_verbose_cnt to non-zero value to enable diff verbose')
        return False

    # for i in range(len):
    #     if diff_point[i] == True:
    #         if err_cnt <= min(diff_num, verbose_cnt):
    #             color_print(False, 'diff at %d: %f vs %f, %f' % (i, a[i], b[i], a[i] - b[i]))
    #         err_cnt += 1
    #     else:
    #         color_print(True, '                                  same at %d: %f vs %f, %f' % (i, a[i], b[i], a[i] - b[i]))
    #         pass

    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python3 fp_diff.py <file1> <file2> [abs_diff_thresh] [rel_diff_thresh] [verbose_cnt]')
    else:
        data1 = np.fromfile(sys.argv[1], dtype=np.uint16)
        data2 = np.fromfile(sys.argv[2], dtype=np.uint16)

        data1 = bfloat16_to_float32(data1)
        data2 = bfloat16_to_float32(data2)

        # data1 = np.fromfile(sys.argv[1], dtype=np.float32)
        # data2 = np.fromfile(sys.argv[2], dtype=np.float32)

        abs_diff_thresh = 0.1#float(sys.argv[3]) if len(sys.argv) >= 4 else 1e-6
        rel_diff_thresh = 0.1#float(sys.argv[4]) if len(sys.argv) >= 5 else 1e-6
        verbose_cnt = int(sys.argv[5]) if len(sys.argv) >= 6 else 10
        print(abs_diff_thresh)
        fp_diff(data1, data2, abs_diff_thresh, rel_diff_thresh, verbose_cnt)

        # print(data1)
