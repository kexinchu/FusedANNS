import numpy as np
import struct

# ==================== 配置路径 (请确认路径正确) ====================
path_query_modal2 = "indexing_and_search/doc/dataset/celeba/test/celeba_modal2_query.ivecs"
path_base_modal2 = "indexing_and_search/doc/dataset/celeba/test/celeba_modal2_base.ivecs"

# 63 个 Case 的完整列表 (QID, Target_ID_Correct, Fused_ID_Wrong)
# 注意：这里使用 Target(Correct) 列作为真正的 GT 来比较属性
cases = [
    (221, 221, 14442), (421, 421, 5679), (495, 495, 1868), (749, 4979, 16498),
    (1145, 1145, 18246), (1437, 4182, 1030), (1616, 6458, 1616), (2189, 2189, 10484),
    (2888, 2631, 2888), (4630, 4630, 13063), (4870, 4870, 12926), (5028, 5028, 13236),
    (5115, 16473, 5115), (5263, 5263, 1168), (6397, 6022, 8612), (6641, 16166, 6641),
    (6938, 6938, 18704), (7017, 9852, 7017), (7285, 7285, 12567), (8240, 8240, 2986),
    (8264, 8264, 4086), (8503, 8503, 1207), (8713, 8713, 2028), (9168, 9168, 9867),
    (9183, 9183, 19560), (9428, 9428, 10795), (10904,
                                               10904, 882), (11477, 11477, 16208),
    (11718, 11718, 15524), (11977, 11977,
                            19498), (12204, 12204, 4418), (13011, 2205, 18886),
    (13357, 13357, 8283), (13424, 11301,
                           13424), (13451, 13451, 19099), (13655, 13655, 887),
    (13670, 17136, 12906), (13971, 13971,
                            2865), (14174, 14174, 1806), (14282, 14282, 3856),
    (14605, 14605, 4279), (14958, 14958, 3568), (15197,
                                                 19465, 15197), (15234, 15234, 18189),
    (15268, 15268, 12194), (15272, 4079, 6766), (15283,
                                                 18034, 8023), (15965, 15965, 14762),
    (16065, 16065, 3088), (16339, 16261,
                           15850), (16416, 16416, 9768), (16465, 16465, 6430),
    (16599, 16599, 19848), (16825, 16825, 2715), (17038,
                                                  15549, 14336), (17625, 17625, 10735),
    (17814, 17814, 16871), (18254, 18254,
                            5545), (18292, 18292, 12251), (18961, 18961, 8604),
    (19340, 19340, 2657), (19598, 9162, 16136), (19639, 19639, 13896)
]


def read_ivecs_row(filename, row_idx):
    with open(filename, 'rb') as f:
        f.seek(0)
        d = struct.unpack('i', f.read(4))[0]
        row_size = 4 + d * 4
        f.seek(row_idx * row_size)
        d_check = struct.unpack('i', f.read(4))[0]
        vec = np.fromfile(f, dtype=np.int32, count=d)
    return vec


# 统计计数器
type_a_count = 0  # 属性干扰 (Fused 属性分更高)
type_b_count = 0  # 路径偏移 (Fused 属性分更低或相等，但还是被选了)

print(f"{'QID':<6} | {'GT_ID':<6} | {'Wrong_ID':<8} | {'GT_Attr':<7} | {'Wrong_Attr':<10} | {'Type'}")
print("-" * 70)

best_case_a = None  # 最典型的 Type A
best_case_b = None  # 最典型的 Type B

for qid, gt_id, wrong_id in cases:
    vec_q = read_ivecs_row(path_query_modal2, qid)
    vec_gt = read_ivecs_row(path_base_modal2, gt_id)
    vec_wrong = read_ivecs_row(path_base_modal2, wrong_id)

    score_gt = np.dot(vec_q, vec_gt)
    score_wrong = np.dot(vec_q, vec_wrong)

    diff = score_wrong - score_gt

    if diff > 0:
        cat = "Type A (Attr Noise)"
        type_a_count += 1
        # 找差距最大的作为典型
        if best_case_a is None or diff > (best_case_a['diff']):
            best_case_a = {'qid': qid, 'gt': score_gt,
                           'wrong': score_wrong, 'diff': diff}
    else:
        cat = "Type B (Path Dev.)"
        type_b_count += 1
        # 找 GT 属性分很高但依然被错过的作为典型
        if best_case_b is None or score_gt > (best_case_b['gt']):
            best_case_b = {'qid': qid, 'gt': score_gt,
                           'wrong': score_wrong, 'diff': diff}

    print(f"{qid:<6} | {gt_id:<6} | {wrong_id:<8} | {score_gt:<7} | {score_wrong:<10} | {cat}")

print("-" * 70)
print(f"Summary: Type A (Attribute Noise) = {type_a_count}")
print(f"Summary: Type B (Graph Path Deviation) = {type_b_count}")
print("-" * 70)
print(
    f"👑 Best Type A Case (For Report): QID {best_case_a['qid']} (Wrong score is {best_case_a['diff']} higher)")
print(
    f"👑 Best Type B Case (For Report): QID {best_case_b['qid']} (GT score is {best_case_b['gt']}, but lost!)")
