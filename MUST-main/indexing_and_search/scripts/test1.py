import numpy as np
import struct

# ==================== 配置路径 ====================
# 你的属性向量文件路径
path_query_modal2 = "indexing_and_search/doc/dataset/celeba/test/celeba_modal2_query.ivecs"
path_base_modal2 = "indexing_and_search/doc/dataset/celeba/test/celeba_modal2_base.ivecs"

# 填入你刚才找到的几个典型 Case (QID, Target_ID, Fused_Wrong_ID)
# 格式: (QID, Target_ID(Correct), Fused_ID(Wrong))
cases = [
    (221, 221, 14442),
    (421, 421, 5679),
    (495, 495, 1868),
    (749, 4979, 16498)
]
# ================================================


def read_ivecs_row(filename, row_idx):
    """读取 ivecs 文件中指定行的数据"""
    with open(filename, 'rb') as f:
        # 读取维度 d
        f.seek(0)
        d_bytes = f.read(4)
        if not d_bytes:
            return None
        d = struct.unpack('i', d_bytes)[0]

        # 计算偏移量: 每一行占用 (4 + d*4) 字节
        row_size = 4 + d * 4
        offset = row_idx * row_size

        # 跳转并读取
        f.seek(offset)
        d_check = struct.unpack('i', f.read(4))[0]
        assert d == d_check, "Dimension mismatch!"

        vec = np.fromfile(f, dtype=np.int32, count=d)
    return vec


print(f"{'Role':<10} | {'ID':<6} | {'Attributes (First 20 dims)...'}")
print("-" * 60)

for qid, target_id, fused_id in cases:
    print(f"\n=== Analyzing Case QID: {qid} ===")

    # 1. 获取 Query 属性
    vec_q = read_ivecs_row(path_query_modal2, qid)

    # 2. 获取 Target (正确结果) 属性
    vec_t = read_ivecs_row(path_base_modal2, target_id)

    # 3. 获取 Fused (错误结果) 属性
    vec_f = read_ivecs_row(path_base_modal2, fused_id)

    # 计算属性匹配度 (Hamming Distance / Overlap)
    # 假设属性是 0/1 二值，点积越高越匹配
    score_t = np.dot(vec_q, vec_t)
    score_f = np.dot(vec_q, vec_f)

    print(f"{'Query':<10} | {qid:<6} | {vec_q[:15]} ...")
    print(
        f"{'Target(C)':<10} | {target_id:<6} | {vec_t[:15]} ... (Match Score: {score_t})")
    print(
        f"{'Fused(W)':<10} | {fused_id:<6} | {vec_f[:15]} ... (Match Score: {score_f})")

    if score_f > score_t:
        print("💡 结论: 融合结果(Fused)的属性匹配分更高！这就是为什么它被选中的原因。")
        print("   (Target 虽然是对的人，但属性匹配度低，被固定权重的融合策略淘汰了。)")
    else:
        print("❓ 结论: 属性分没有更高，可能是视觉特征(Modal 1)被某种方式干扰了。")
