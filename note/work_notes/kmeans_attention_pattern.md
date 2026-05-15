# K-means + Attention：处理超长序列/超大集合的范式

> **场景**：当你想对一个集合做 attention，但集合规模 N 太大、算不动时。
>
> **核心 idea**：先 K-means 聚成 k 个代表向量（centroid），在 k 个 centroid 上做 attention，复杂度从 `O(N · h²)` 降到 `O(k · h²)`，**通常 k=20 就够，压缩比 50000+ 倍**。
>
> **本质上**：用"群体代表"替代"个体全量"，在表征精度和计算可行性之间做取舍。

---

## 这个范式的来源

- 直接来自 **RALM (KDD 2019, Tencent)** 的 seeds clustering 设计
- 详见 `paper/sequence_modeling/short/note/RALM.md` 4.3 / 5.3 节

RALM 的具体应用：百万级 seed users → 聚成 k=20 个 cluster centroid → 在 centroid 上做 global/local attention。

---

## 与其他长序列处理方法的对比

业界处理"长序列 attention"主要有三条路径：

| 路径               | 代表方法               | 思路                                            | 优劣                              |
|------------------|--------------------|-----------------------------------------------|---------------------------------|
| **Top-K 检索**     | SIM、ETA、TWIN       | 用 hash / SimHash 找和 target 最相关的 K 个，丢弃其他      | **保留高相关、丢弃低相关**，适合"target 明确"场景 |
| **聚类压缩**         | RALM、本文            | K-means 聚成 k 个 centroid，**保留全量信息**（以压缩形式）     | **保留群体结构**，适合"想保留所有信号"场景        |
| **稀疏 attention** | Longformer、BigBird | sliding window / random / global attention 组合 | NLP 范式，推荐里少见                    |

**核心区别**：Top-K 是**信息丢弃**，聚类压缩是**信息聚合**。

> **Top-K 适合**：你知道 target 是谁，只关心和它相关的部分（如 CTR 预估，target item 明确）
> **聚类压缩适合**：你不知道 target，或者想保留全集信息（如群体表征、多兴趣建模）

---

## 在 User-Sequence Search Ranking 中的可能应用

### 痛点对照

User-sequence 在 search ranking 里典型问题：
- 用户历史行为序列可能几百到几千条
- DIN/DIEN 风格的 target attention 在长序列上 latency 不可控
- SIM/ETA 走 Top-K 检索，**丢掉了大部分行为信号**

### 方案：聚类后做 target attention

```
用户行为序列 [item_1, item_2, ..., item_N]   N = 几千
        ↓ K-means clustering (基于 item embedding)
k 个 cluster centroids   k = 20~50
        ↓ target-aware attention (query = 当前 query 或 candidate item)
单个 user 表征向量
        ↓ 喂给 ranking 模型
```

**对比 SIM/ETA**：
- SIM/ETA：从 N 条中选 K 条相似的 → **丢掉 N-K 条**
- 聚类方案：把 N 条压成 k 个 centroid → **N 条信息都参与了，只是被聚合了**

### 这个方案适合什么场景

✅ **适合**：
- 用户兴趣**多样**（聚类才有意义）
- query 在多个兴趣类目间切换（需要不同时刻激活不同 cluster）
- 想保留长尾兴趣信号（Top-K 容易丢长尾）

❌ **不适合**：
- 用户兴趣单一（聚类没区分度，等于平均池化）
- 极短序列（N < 50，没必要聚类）
- target 极其明确（Top-K 更直接）

### 与 Top-K 路线的组合

两条路径可以**组合使用**：

```
长序列 → SIM 粗筛 top-100 条 → K-means 聚成 k=10 个 centroid → attention
        └─ 丢掉无关  ──────┘  └─ 聚合相关 ─────────────────────┘
```

这样既**砍掉了长尾噪声**（Top-K 的作用），又**保留了相关部分的结构**（聚类的作用）。

---

## 实现要点

### 1. 聚类用什么 embedding

- **用 item embedding 直接聚** ← 最简单，常用
- 也可以用 (item × position × time) 拼接后的向量聚（保留时序信号）
- **不要用 raw feature 聚**（embedding 已经过学习，结构更好）

### 2. k 怎么选

- 论文里 RALM 用 k=20，elbow point
- 用户序列场景建议 **k=10~50**
- 太小 → 聚类粒度粗，丢信息
- 太大 → 计算优势减弱
- **做 grid search 找 elbow point**

### 3. 聚类频率

| 场景                      | 重聚类频率                    |
|-------------------------|--------------------------|
| **训练时**                 | 每个 epoch 开始重聚类（RALM 做法）  |
| **推理时**（用户级聚类）          | 离线天级更新                   |
| **推理时**（per-request 聚类） | 实时 K-means，但要控制 k 不超过 50 |

> 用户级聚类可以**预计算缓存**——每个用户 → k 个 centroid 存起来，request 时直接 fetch。

### 4. 训练时的 iterative 注意事项

- 如果 item embedding 在训练过程中**也在更新**（端到端训练）
- 那么聚类结果会随训练漂移 → 需要**周期性重聚类**
- 否则 centroid 过时，attention 计算的就是过期的"群体"

> **本质是 EM 思路**：E 步聚类，M 步训练 embedding，交替进行。

### 5. 不要漏掉 mask

- 聚类结果中某些 cluster 可能为空（极端情况）
- attention 时要 mask 掉空 cluster
- 否则 softmax 会把权重分给空 cluster，浪费容量

---

## 复杂度账（带具体数字）

假设 user-sequence 任务：N=2000, h=128（embedding 维度）, B=512（batch size）

| 方案                             | 单 sample attention 复杂度          | Batch 计算量 | 备注             |
|--------------------------------|---------------------------------|-----------|----------------|
| 全量 target attention            | `O(N·h)` = 2000 × 128 ≈ 2.5×10⁵ | 1.3×10⁸   | 太慢             |
| SIM/ETA (top-K=100)            | `O(K·h)` = 100 × 128 ≈ 1.3×10⁴  | 6.6×10⁶   | 快但丢信息          |
| **K-means + attention (k=20)** | `O(k·h)` = 20 × 128 ≈ 2.6×10³   | 1.3×10⁶   | **快 + 保留全量信息** |
| 组合 (SIM 100 + K-means 10)      | `O(k·h)` = 10 × 128 ≈ 1.3×10³   | 6.6×10⁵   | 最快             |

> **关键观察**：聚类方案比 SIM 还快，而且保留了全量信息——这是个被低估的方案。

---

## 这个范式的迁移地图

| 场景                             | 套用方式                                              |
|--------------------------------|---------------------------------------------------|
| **User long-history sequence** | item embedding 聚类，target attention 在 centroid 上 ⭐ |
| **群体/look-alike 建模**           | user embedding 聚类（RALM 原始场景）                      |
| **多兴趣建模**                      | 用聚类天然得到 k 个"兴趣中心"                                 |
| **大候选集召回粗排**                   | item embedding 聚类，query 在 centroid 上做粗排           |
| **MoE expert 数太多**             | 聚类相似 expert，让一组 expert 共享参数                       |
| **超长文档检索**                     | 段落 embedding 聚类，query 先匹配 cluster 再细化             |

---

## 关键 takeaway

1. **当 N 太大算不动 attention 时**，K-means 聚成 k 个代表是几乎总值得一试的方案
2. **聚类是"信息聚合"**，Top-K 是"信息丢弃"——两者哲学不同，可组合使用
3. **k=20 通常够用**（RALM 的实验结论），不需要追求大 k
4. **训练时记得 iterative**——embedding 在变，聚类也要跟着变
5. **优先用学过的 embedding 聚类**，而不是 raw feature

---

## 待验证假设（针对我的 user-sequence project）

- [ ] 当前 user-sequence 长度分布是什么样？长尾占比多少？
- [ ] 如果对用户行为做 K-means，k=20 时类间距离/类内距离是什么量级？
- [ ] vs SIM/ETA 的离线效果对比：哪个 AUC 更高？哪个 latency 更低？
- [ ] 训练时每个 epoch 重聚类的额外开销能接受吗？
- [ ] 聚类后的 centroid 是否还能保留时序信号？（这可能是个弱点）

---

## 相关笔记

- `paper/sequence_modeling/short/note/RALM.md` —— 范式来源
- `paper/sequence_modeling/long/note/ETA-Net.md` —— Top-K 路线代表
- `paper/sequence_modeling/long/note/MIMN.md` —— 记忆槽方式（另一种"固定大小存储"思路）
- `dcn_expert_split_thinking.md` —— 工作场景思考