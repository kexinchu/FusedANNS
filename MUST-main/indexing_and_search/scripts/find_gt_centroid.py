import numpy as np
import struct

# ================= 配置路径 =================
path_base_visual = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba/test/resnet50_encode/celeba_modal1_base.fvecs"
path_centroids_v = "/Users/mac/github/FusedANNS/MUST-main/indexing_and_search/doc/dataset/celeba_centroids/celeba_centroids_visual.fvecs"

# King Case
gt_id = 18254


def read_fvecs(filename, idx=None):
    with open(filename, 'rb') as f:
        d = struct.unpack('i', f.read(4))[0]
        if idx is not None:
            f.seek(idx * (4 + d * 4))
            return np.fromfile(f, dtype=np.float32, count=d)
        f.seek(0)
        all_vecs = []
        while True:
            bytes_d = f.read(4)
            if not bytes_d:
                break
            vec = np.fromfile(f, dtype=np.float32, count=d)
            all_vecs.append(vec)
        return np.array(all_vecs)


print(f"Loading data for GT ID {gt_id}...")
gt_vec = read_fvecs(path_base_visual, idx=gt_id)
centroids = read_fvecs(path_centroids_v)

# 计算 GT 到所有簇中心的距离
dists = np.linalg.norm(centroids - gt_vec, axis=1)
best_centroid_for_gt = np.argmin(dists)
min_dist = dists[best_centroid_for_gt]

print("-" * 40)
print(
    f"Ground Truth (ID {gt_id}) is closest to Centroid: {best_centroid_for_gt}")
print(f"Distance to Centroid: {min_dist:.4f}")
print("-" * 40)
print(f"Target-Only Strategy picked: 0")
print(f"Adaptive Strategy picked:    20")
print("-" * 40)

if best_centroid_for_gt == 20:
    print("✅ GREAT NEWS: Adaptive picked the correct cluster! Just increase L_search.")
elif best_centroid_for_gt == 0:
    print("❌ BAD NEWS: Adaptive moved away from the correct cluster.")
else:
    print(
        f"⚠️ TRICKY: GT is in Cluster {best_centroid_for_gt}, but we picked 0 or 20.")
    print("   Solution: We need Multi-Entry Points (Top-K).")
