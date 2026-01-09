import os

# ================= 配置路径 =================
# 1. Target Only (基准, D2=0, 82.8%)
file_target_only = "indexing_and_search/doc/result/2026_01_06/celeba_MSAG_res_w1_-0.1000_w2_-1.0000.txt"

# 2. Fused (融合, D2正常, 87.6%)
file_fused = "indexing_and_search/doc/result/2026_01_06/celeba_MSAG_res_w1_-0.0848_w2_-1.1855.txt"
# ===========================================


def load_data_map(path):
    """
    加载文件并返回一个字典:
    { qid: {'is_correct': bool, 'retrieved_id': int, 'gt_id': int} }
    """
    data_map = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue

                # 解析格式: qid, is_correct, retrieved_id, gt_id
                qid = int(parts[0].strip())
                is_correct = int(parts[1].strip())
                ret_id = int(parts[2].strip())
                gt_id = int(parts[3].strip())

                data_map[qid] = {
                    'is_correct': bool(is_correct),
                    'retrieved_id': ret_id,
                    'gt_id': gt_id
                }
        return data_map
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


# 1. 加载详细数据
data_target = load_data_map(file_target_only)
data_fused = load_data_map(file_fused)

if not data_target or not data_fused:
    print("Error: Could not load data files.")
    exit(1)

# 2. 找出 "Broke by Fusion" 的 Query
# 定义: Target 算对了 (True) AND Fused 算错了 (False)
broken_queries = []

# 为了安全起见，取两个文件 QID 的交集进行遍历
all_qids = set(data_target.keys()) & set(data_fused.keys())

for qid in all_qids:
    res_t = data_target[qid]
    res_f = data_fused[qid]

    # 逻辑判断: M1 的证据 (Target对，Fused错)
    if res_t['is_correct'] and not res_f['is_correct']:
        broken_queries.append({
            'qid': qid,
            'gt_id': res_t['gt_id'],
            'target_ret': res_t['retrieved_id'],  # 这是对的
            'fused_ret': res_f['retrieved_id']   # 这是错的
        })

# 3. 输出结果
print(f"Total Queries Checked: {len(all_qids)}")
print(f"Queries Broke by Fusion (Count): {len(broken_queries)}")
print("-" * 80)
print(f"{'QID':<10} | {'GT ID':<10} | {'Target(Correct)':<18} | {'Fused(Wrong)':<15}")
print("-" * 80)

# 排序输出 (按 QID 从小到大)
broken_queries.sort(key=lambda x: x['qid'])

for item in broken_queries:
    print(
        f"{item['qid']:<10} | {item['gt_id']:<10} | {item['target_ret']:<18} | {item['fused_ret']:<15}")

print("-" * 80)
print("提示: 这些 Query 就是你在论文 Case Study 中需要分析的例子。")
print("它们证明了辅助模态(Modal 2)在这些特定情况下引入了噪声或权重过大导致检索偏离。")
