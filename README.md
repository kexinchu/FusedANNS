# 用于 MSTM/目标模态搜索的查询自适应路由

## Problem
- MSTM in Vector DB
目标模态多模态搜索 (MSTM)：给定一个大规模数据集合$D$，每个对象由多模态表示${x^{(T)}, x^{(A_1)}, ...}$，其中目标模态 $T$（如image/video）是我们最终要返回的对象类型；而对于查询$q$，它包含目标模态$q^{(T)}$ 和 若干辅助模态 $q^{(A)}$（如文本描述，属性，上下文），目标是返回top-k个目标模态对象匹配 "组合意图"

- 现状
    - 方法1，融合成一个“固定相似度”再检索
        - 好处：可索引、可扩展
        - 代价：相似度表达受限（通常线性、全局权重），更关键的是 融合策略固定，无法针对 query 的需求变化。
    - 方法2，分模态检索 + rerank/filter
        - 好处：表达更强
        - 代价：index存储成本 + 多次检索、候选集合大、代价不可控，难以在高并发/低延迟下稳定运行。
    - 核心问题：MSTM 的“组合意图”是 query-specific 的（有些 query 更依赖目标模态，有些更依赖辅助模态约束），但现有 graph-based ANN 检索通常是“固定的 traversal 策略 + 固定的相似度”。

##  Motivation
- 不再把辅助模态硬塞进“融合距离”，而是把它提升为 Search Controller：
    - 索引仍然只在目标模态空间建图$G_T$（HNSW/DiskANN/NSG 都行）
    - 辅助模态不改变索引结构，而是动态影响：
        - Neighbor扩展门控（扩展哪些邻居（优先级）、扩展多少）
        - entry points 选择
        - 搜索预算分配与 early stopping
- 核心设计：
    - ANNS search控制 (neighbor拓展gate + priority)
        - Priority：对候选节点$v$，除了目标相似度$s_T(s^{(T), v})$，计算一个轻量的辅助匹配信号$g(q^{(A)}, v)$ (来自节点的signature)： $priority(v) = s_T(q^{(T)}, v) + λ * g(q^{(A)}, v)$；然后判断priority来选择neighbors。（λ是需要训练的到的缩放参数, g可以是distance fucntion）
        - 单一的λ + g，系统就接近融合搜索场景 -> $λ = w_λ * {q^{(T)}, q^{(A)}}$;  
        - Note：离线graph-based Index构建时候需要有一定的冗余，因为目前的base graph连通性与实际计算的连通性并不完全对等
        - 缺陷：图连通性 与 search路径的 mismatch，导致搜索精度下降
    - Entry Point -> 多入口 + gating
        - 不同于HNSW/DiskANN的seed随机入口，这里因为 $λ = w_λ * {q^{(T)}, q^{(A)}}$ 是变化的，为了避免局部最优解 + 加速搜索 -> 多入口 + gating
        - 多入口 sets： 对targte modality + 辅助motality 进行sample + 聚类，记录簇中心 $c_j^{(A)}$为 centroid table
        - gating：计算 $priority(v) = s_T(q^{(T)}, v) + λ * g(q^{(A)}, v)$ 从centroid table中获取入口
    - Search 预算分配 + early stop
        - 难query获得更多预算
        - early stop：同时满足
            - 目标模态收敛：$\Delta s_T<ϵ_T$
            - 约束模态收敛：$\Delta g<ϵ_A$ 或 topK 的 $g$ 达到阈值 $g≥τ$

- Challenges
    - Graph 连通性 ≠ Query-Dependent 可达性？
        - 传统 HNSW / DiskANN 的图连通性是针对单一距离函数设计的，但是这里的设计当 λ 改变时，最优路径跟图中的边分布不一定match；在原本$s_T$下 “远” 的节点，可能在 priority 空间下是关键中转
    - Entry Point 对 Query-Adaptive Search 极度敏感
    - 低成本的 预算分配与 Early Stop，以及两者的可靠性

## Motivation Tests

- M1. **MSTM 存在“机制切换”**；证明“单一融合距离 / 固定 λ 是不稳定的”。：
    - 固定graph，固定λ
    - 检索相同target modality + diff 辅助模态
    - 对比结果

- M2. 不同entry point的影响；对比下面三种
    - Random seed（HNSW baseline）
    - Target-only best centroid
    - Target + Auxiliary gated centroid

- M3. 不同 query 对 search预选/early stop的收益差异
    - 对每个 query：逐步增加 budget；记录：
        - best $s_T$
        - best $g$
        - best $priority$