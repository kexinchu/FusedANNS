import numpy as np
import struct

# ================= 配置路径 =================
path_query_visual = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba/test/resnet50_encode/celeba_modal1_query.fvecs"
path_query_attr   = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba/test/celeba_modal2_query.ivecs"
path_centroids_v = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba_centroids/celeba_centroids_visual.fvecs"  # 假设你生成了这个
path_centroids_a  = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba_centroids/celeba_centroids_attr.ivecs"   # 假设你生成了这个

# King Case
target_qid = 18254
# 假设 weights (注意符号，这里模拟“距离”，越小越好)
w1 = 0.08  # 对应 -0.08
w2 = 1.18  # 对应 -1.18

def read_fvecs(filename, idx=None, count=None):
    # 简化版读取，实际请复用你之前的读取函数
    with open(filename, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]
        if idx is not None:
            f.seek(idx * (4 + d * 4))
            return np.fromfile(f, dtype=np.float32, count=d)
        # 读取全部用于 centroids
        f.seek(0)
        all_vecs = []
        while True:
            bytes_d = f.read(4)
            if not bytes_d: break
            vec = np.fromfile(f, dtype=np.float32, count=d)
            all_vecs.append(vec)
        return np.array(all_vecs)

def read_ivecs(filename, idx=None):
    with open(filename, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]
        if idx is not None:
            f.seek(idx * (4 + d * 4))
            return np.fromfile(f, dtype=np.int32, count=d)
        f.seek(0)
        all_vecs = []
        while True:
            bytes_d = f.read(4)
            if not bytes_d: break
            vec = np.fromfile(f, dtype=np.int32, count=d)
            all_vecs.append(vec)
        return np.array(all_vecs)

# 1. 加载数据
q_v = read_fvecs(path_query_visual, idx=target_qid)
q_a = read_ivecs(path_query_attr, idx=target_qid)
cents_v = read_fvecs(path_centroids_v) # shape [256, 2048]
cents_a = read_ivecs(path_centroids_a) # shape [256, 40]

print(f"Analyzing Entry Points for QID {target_qid}...")

# 2. 模拟 Strategy 1 (Target Only)
dists_v = np.linalg.norm(cents_v - q_v, axis=1) # L2 distance
best_ep_1 = np.argmin(dists_v)
print(f"Strategy 1 (Target-Only) chose Centroid ID: {best_ep_1} (Dist: {dists_v[best_ep_1]:.4f})")

# 3. 模拟 Strategy 2 (Adaptive)
# 注意归一化！这里假设简单的加权距离
# 这里的 dists_a 需要根据你的 metric 来（Hamming 或 L2）
# 假设是 L2 用于演示
dists_a = np.linalg.norm(cents_a - q_a, axis=1) 

# 组合距离
combined_scores = w1 * dists_v + w2 * dists_a
best_ep_2 = np.argmin(combined_scores)
print(f"Strategy 2 (Adaptive)    chose Centroid ID: {best_ep_2} (Score: {combined_scores[best_ep_2]:.4f})")
print(f"   -> Visual Dist: {dists_v[best_ep_2]:.4f}")
print(f"   -> Attr Dist:   {dists_a[best_ep_2]:.4f}")

if best_ep_1 == best_ep_2:
    print("❌ CONCLUSION: Both strategies picked the SAME centroid! Increase w2.")
else:
    print("✅ CONCLUSION: Strategies picked DIFFERENT centroids!")