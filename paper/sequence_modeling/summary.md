# User sequence paper summary

## Each paper short summary

## 1. DIN (Deep Interest Network)
- 动机：
  - 传统的Embedding & Pooling方法将用户所有历史行为压缩为固定向量，丢失了用户兴趣的多样性（Diversity）。用户对不同目标商品（Target Item）的兴趣点应该是不同的。
- 核心贡献：
  - 引入了Local Activation Unit（局部激活单元）。
  - 自适应激活函数 Dice。
- 模型总结：
  - 不把历史行为一视同仁，而是根据当前的候选广告/商品，计算历史行为中每个Item与当前Target的权重（Attention Score），进行加权求和。
- 用户行为属性信息类型：
  - Item ID + Category ID + Shop ID 等基础属性。
- 用户行为是否是多种行为：
  - 否（通常仅基于点击）。
- 用户行为是否是多种序列：
  - 否（通常为一个单一行为序列），不过可以扩展到多序列作为兴趣提取器。
- 用户行为是否有负反馈建模：
  - 否（只有点击过的item）。
- 序列长度：
  - 中短序列（通常截取最近N个，如50-100）。
- 可继续拓展研究方向：
  - 如何降低Attention计算复杂度以处理更长的序列。
  - 结合时间衰减因素。
  - 多兴趣扩展
  - 与序列建模结合

## 2. DIEN (Deep Interest Evolution Network)
- 动机：
  - DIN捕捉了兴趣的相对强度，但忽略了兴趣随时间的动态演化（Evolution），即“当前的兴趣是由过去的兴趣演变而来的”。
- 核心贡献：
  - 提出了兴趣抽取层和兴趣演化层，设计了AUGRU（Attention Update Gate）。
  - 设计辅助损失，即使用下一个行为来监督当前的隐藏状态，强化兴趣抽取层的表征学习。
- 模型总结：
  - 先用GRU抽取序列的基础兴趣，再利用Target Attention机制控制GRU的更新门（Update Gate），使得模型只让与目标相关的兴趣随着时间传递下去。
- 用户行为属性信息类型：
  - 带有时间顺序的Item ID + Category ID + Shop ID序列。
- 用户行为是否是多种行为：
  - 否。
- 用户行为是否是多种序列：
  - 否。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 中短序列 （50-100）。
- 可继续拓展研究方向：
  - 更复杂的序列建模（如Transformer）
  - 兴趣演化方向的控制与预测

## 3. DSIN (Deep Session Interest Network)
- 动机：
  - 用户行为天然具有**Session（会话）**结构。Session内部行为高度同质，Session之间行为异质。简单的序列模型忽略了这种内在结构。
- 核心贡献：
  - 对序列进行Session切分，建模Session内和Session间的兴趣。
- 模型总结：
  - 使用Transformer（Self-Attention）捕捉Session内的上下文兴趣，使用Bi-LSTM捕捉Session间的兴趣演化过程。最后，同DIN计算每个会话兴趣的重要性。
- 用户行为属性信息类型：
  - 带有时间戳的点击序列（用于切分Session，如30分钟间隔）。每个行为包含：Item ID, Category ID, Shop ID, timestamps, behavior type。
- 用户行为是否是多序列：
  - 否。
- 用户行为是否是多种序列：
  - 否。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 中短序列（由多个短Session组成的序列）。
- 可继续拓展研究方向：
  - Session内的多行为交互（如点击后立即加购）
  - 动态Session划分
  - Session意图识别
  - 跨Session长期依赖建模

## 4. DMT (Deep Multifaceted Transformers)
- 动机：
  - 在电商场景中，用户有点击、加购、下单等多种行为。不同行为反映了不同强度的偏好。需同时优化多目标（CTR/CVR），且用户有多种行为
- 核心贡献：
  - 多行为建模 + MMoE架构 + 偏置网络建模位置/邻近偏差。
- 模型总结：
  - 使用多个独立的Transformer Encoder分别处理不同类型的行为序列（Click, Cart, Order），用 Decoder 来表示用户关于目标商品的兴趣向量，结合Bias Encoding区分位置、邻近上下文等带来的偏差，最后通过MMoE进行多任务（CTR/CVR）预测。
- 用户行为属性信息类型：
  - Item ID + Category ID + Brand ID + Shop ID
- 用户行为是否是多种行为：
  - 是（点击、加购、购买）
- 用户行为是否是多种序列：
  - 是，每种行为一个单独序列。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短 + 中 + 长期
- 可继续拓展研究方向：
  - 稀疏行为（如下单）的增强学习
  - 不同行为之间的因果关系建模
  - 更多行为类型融合

## 5. DSTN (Deep Spatiotemporal Network)
- 动机：
  - 用户的点击不仅受历史行为影响（时间维度），还受当前页面共现的上下文广告影响（空间维度）。
- 核心贡献：
  - 时空联合建模。
  - 显式建模上下文广告信息。
- 模型总结：
  - 模型输入不仅包含用户历史点击和历史未点击序列，还包含当前页面展示的其他广告序列。设计了专门的交互层融合Target、Context Ads和History。
  - 对每个辅助数据，计算 Interactive Attention
- 用户行为属性信息类型：
  - Item ID + Category ID + Shop ID
- 用户行为是否是多种行为： 
  - 否
- 用户行为是否是多种序列：
  - 是，历史点击序列 + 历史未点击序列 + 当前页面曝光序列。三种辅助数据引入。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短
- 可继续拓展研究方向：
  - 页面布局（Layout）信息的引入
  - 更复杂的图神经网络建模空间关系
  - 上下文广告的显式竞争/互补关系建模

## 6. DHAN (Deep Hierarchical Attention Network)
- 动机：
  - 用户的兴趣具有层级结构（例如：电子产品 -> 手机 -> iPhone，品类->品牌->产品）。扁平的序列建模难以捕捉这种从粗粒度到细粒度的语义。
- 核心贡献：
  - 层次化注意力网络。
  - 构建从具体到抽象的多层兴趣树
- 模型总结：
  - 构建多层Attention结构，底层关注具体的Item上下文，高层对类目（Category）或Session级别的兴趣进行聚合和重加权，逐层抽象用户兴趣。。
- 用户行为属性信息类型：
  - 具有层级关系的属性（Item, Brand, Category），最为每个序列的类型。
- 用户行为是否是多种行为：
  - 否。
- 用户行为是否是多种序列：
  - 是，可以是兴趣类型，类别，价格，品牌，都单独构成一个序列。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短。
- 可继续拓展研究方向：
  - 结合知识图谱（KG）的层级结构
  - 自动学习最优的层级划分，发现兴趣层次、多维度层次（价格、品牌等）联合建模。

## 7. HUP (Hierarchical User Profiling)
- 动机：
  - 用户分层兴趣的问题同DHAN
  - 用户行为有细粒度“微行为”，比如：用户与物品组件（如图片、评论）的交互
- 核心贡献：
  - 提出金字塔RNN结构，从微行为->物品->品类分层编码
  - 利用微行为序列（点击、浏览详情等）
- 模型总结：
  - 分层RNN：底层处理微行为序列，中层聚合为物品兴趣，高层聚合为品类兴趣，输出多层次用户画像。
- 用户行为属性信息类型：
  - 物品(vi)、各类别(ci)、行为类型(bi)、停留时间桶(di分桶后)、时间间隔桶(gi分桶后)
- 用户行为是否是多种行为：
  - 否。
- 用户行为是否是多种序列：
  - 是，每一层代表不同的层级兴趣序列。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短。
- 可继续拓展研究方向：
  - 微行为语义的深度理解
  - 微行为的类型

## 8. TiSSA (Time Slice Self-Attention)
- 动机：
  - 难以准确捕捉高度相关的序列行为，且容易受到无关行为（噪声）的干扰
- 核心贡献：
  - 提出时间切片自注意力机制，挖掘时间维度信息
  - 按多尺度时间窗口切片，分别进行切片内和切片间自注意力
- 模型总结：
  - 先会话划分，再按时间切片动态切割，切片内捕捉局部依赖，切片间捕捉全局依赖。
- 用户行为属性信息类型：
  - Item ID + Category ID + Brand ID + Shop ID
- 用户行为是否是多种行为：
  - 否。
- 用户行为是否是多种序列：
  - 否。单一序列进行多层切分计算，类似 DIEN/DSIN
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短。
- 可继续拓展研究方向：
  - 结合周期性时间特征（周末/工作日/节假日）

## 9. BST (Behavior Sequence Transformer)
- 动机：
  - Transformer在NLP表现优异，且能捕捉长距离依赖，可以直接迁移到RecSys替代RNN/CNN。
- 核心贡献：
  - 将标准Transformer Encoder应用于用户行为序列。
  - 强调行为顺序的重要性，更改了位置编码设计
- 模型总结：
  - 将用户行为序列视为一个序列 + 候选物品，用Transformer编码器（带位置编码）提取新序列表征
- 用户行为属性信息类型：
  - Item ID + Category I + 时间间隔
- 用户行为是否是多种行为：
  - 否。
- 用户行为是否是多种序列：
  - 否。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 短（20）。
- 可继续拓展研究方向：
  - 针对超长序列的线性Attention优化
  - BERT式的预训练（Masked Item Prediction），类似BERT4Rec。

## 10. TransAct v1. TransActv2
- 动机：
  - Pinterest等场景下，用户行为包含即时搜索意图和长期浏览偏好。需要一种高效、可扩展的方式处理海量用户的混合序列。
- 核心贡献：
  - 实时-批量混合架构：TransAct(实时Transformer) + 批量用户嵌入
  - 工程上实现低延迟服务优化
- 模型总结：
  - 实时模块处理近期行为，批量模块提供长期画像，两者特征拼接后输入排序模型。
- 用户行为属性信息类型：
  - Pretrained item embeddings + Action Type (Save/Click/Hide)
- 用户行为是否是多种行为：
  - 是（保存Pin、点击、隐藏）。
- 用户行为是否是多种序列：
  - 否。是一个按时间顺序排序的多行为混合序列。
- 用户行为是否有负反馈建模：
  - 是。
- 序列长度：
  - 短 (100)。
- 可继续拓展研究方向：
  - 多模态内容（图片/视频）与行为的端到端融合
  - 超大规模检索架构，不用 pretrain embedding
  - 更精细的实时-长期兴趣融合机制

## 11 TransAct v2
- 动机：
  - 无法利用终身行为序列end-to-end学习
- 核心贡献：
  - 结合实时和终身用户序列（10^4量级）
  - 引入下一动作预测作为辅助任务
  - 工程上配套高效数据流水线与定制Triton内核
- 模型总结：
  - 通过面向候选的NN检索从终身序列中提取相关子序列 + 短期Session（Real-time），同V1用Transformer编码，并加入对比学习的下一动作预测损失。
- 用户行为属性信息类型：
  - Pretrained item embeddings + Action Type (Save/Click/Hide) + Timestamps + Action surface (行为发生界面)
- 用户行为是否是多种行为：
  - 是（保存Pin、点击、隐藏）。
- 用户行为是否是多种序列：
  - 否。是一个按时间顺序排序的多行为混合序列。
- 用户行为是否有负反馈建模：
  - 是。
- 序列长度：
  - 长混合序列（短期全量 + 长期采样 = 192）。
- 可继续拓展研究方向：
  - 多模态内容（图片/视频）与行为的端到端融合
  - 终身序列的无损高效检索
  - 多目标辅助任务设计

## 12. SDM (Sequential Deep Matching)
- 动机：
  - 用户一次会话中可能包含多种不同的兴趣
  - 用户的长期历史行为复杂多样，可能短期存在冲突，短期会话与长期行为应协同
- 核心贡献：
  - 长短期兴趣分离与融合。
- 模型总结：
  - 短期用 LSTM+ multi-head self-attention，长期用 User Profile+Attention。最后通过Gating机制融合两个向量用于召回。
- 用户行为属性信息类型：
  - Item ID + Leaf Category ID + First Level Category ID + Brand ID + Shop ID
- 用户行为是否是多种行为：
  - 否（主要用于序列召回）。
- 用户行为是否是多种序列：
  - 是，长期 + 短期 分开建模。
- 用户行为是否有负反馈建模：
  - 否。
- 序列长度：
  - 混合序列（短期 + 长期）。
- 可继续拓展研究方向：
  - 在精排（Ranking）阶段的应用
  - 如何动态决定长短期的权重。
  - 多兴趣向量的动态数量
  - 会话意图的识别与分解

## 13. MIMN (Practice on Long Sequential User Behavior Modeling)
- 动机：
  - 工业界用户历史累积非常长（如淘宝1000+点击），直接输入模型会导致存储爆炸和延迟过高。
- 核心贡献：
  - 系统解耦：UIC（用户兴趣中心）异步更新
  - 算法创新：基于NTM的记忆网络，以固定内存存储长期兴趣
  - 记忆利用正则化与记忆诱导单元
- 模型总结：
  - 使用外部Memory Network矩阵存储用户兴趣。每当有新行为产生，更新Memory（Write）；预测时直接读取Memory（Read）。实现了计算与预测的解耦。实现零延迟。
- 用户行为属性信息类型：
  - Item ID
- 用户行为是否是多种行为：
  - 否（通常聚焦点击）。
- 序列长度：
  - 超长序列（1000+）。
- 可继续拓展研究方向：
  - Memory的写入策略优化（如何遗忘旧信息）
  - 记忆交互
  - 基于大模型（LLM）的记忆压缩与总结。

## Each paper summary table
| 论文              | 核心动机                     | 核心模型结构                                                                       | 用户行为属性信息类型                                                                  | 是否多行为 | 是否多序列 | 序列长度类型           | 是否负反馈 |
|:----------------|:-------------------------|:-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|:------|-------|:-----------------|-------|
| **DIN**         | 兴趣多样性，不同Target激活不同历史     | - Local Activation Unit（局部激活单元) <br/> - 自适应激活函数 Dice                         | Item ID + Category ID + Shop ID                                             | 否     | 否     | 短序列 （50 - 100）   | 否     |
| **DIEN**        | 兴趣随时间的动态演化               | GRU + AUGRU (Attention Update Gate)                                          | Item ID + Category ID + Shop ID                                             | 否     | 否     | 短 （50 - 100）     | 否     |
| **DSIN**        | 用户行为天然具有Session结构        | Transformer (Session内) + Bi-LSTM (Session间) + Local Activation Unit          | Item ID + Category ID + Shop ID + Timestamps + Behavior Type                | 否     | 否     | 短 (Session粒度)    | 否     |
| **DMT**         | 不同行为(点击/购买)反映了不同强度的偏好    | Multi-Transformer (Encoder + Decoder) + MMoE + Bias Encoding                 | Item ID + Category ID + Brand ID + Shop ID                                  | **是** | **是** | 短 + 中 + 长期       | 否     |
| **DSTN**        | 空间(上下文广告)与时间(历史)共同影响     | Interactive Attention (候选item + 辅助数据注意力)                                     | Item ID + Category ID + Shop ID                                             | 否     | **是** | 短                | 是     |
| **DHAN**        | 兴趣具有层级结构(类目/品牌)          | Hierarchical Attention (多层注意力，每一层是DIN的思路)                                    | Item ID，Category ID，Brand ID                                                | 否     | **是** | 短                | 否     |
| **HUP**         | 用户分层兴趣和用户行为有细粒度“微行为”     | Pyramid RNN (Behavior-LSTM Cell)                                             | Item ID + Category ID + Shop ID + Behavior Type + dWell + Timestamps bucket | 否     | **是** | 短                | 否     |
| **TiSSA**       | 高度相关的行为，且容易受到无关行为（噪声）的干扰 | Time-GRU + Inter Slice Self Attention + Hierarchical Self-Attention          | Item ID + Category ID + Brand ID + Shop ID                                  | 否     | 否     | 短                | 否     |
| **BST**         | 迁移NLP强特征提取能力             | Standard Transformer Encoder + time interval position embedding              | Item ID + Category I + 时间间隔                                                 | 否     | 否     | 短 (20)           | 否     |
| **TransAct v1** | 工业界大规模场景兼顾实即时搜索意图和长期浏览偏好 | Standard Transformer Encoder (append candidate) + short long term fusion     | Pretrained item embeddings + Action Type                                    | **是** | 否     | 短 (100)          | 是     |
| **TransAct v2** | 无法利用终身行为序列end-to-end学习   | Hybrid Transformer (Short-term + Sampled Long-term) + short long term fusion | Pretrained item embeddings + Action Type + Timestamps + Action surface      | **是** | 否     | 长混合序列            | 是     |
| **SDM**         | 长短期兴趣不一致甚至冲突             | LSTM + Multi-head self-attention (Short) + Attention (Long) + Gating         | Item ID + Leaf Category ID + First Level Category ID + Brand ID + Shop ID   | 否     | **是** | 长混合序列            | 否     |
| **MIMN**        | 超长序列带来的存储与延迟瓶颈           | Memory Network (NTM) + UIC (User Interest Center)                            | Item ID                                                                     | 否     | 否     | **超长序列** (1000+) | 否     |

## All papers group summary

| 分类流派       | 论文           | 核心演化点 (Key Innovation)      | 模型关键组件                                | 序列特征            | 多行为   |
|:-----------|:-------------|:----------------------------|:--------------------------------------|:----------------|:------|
| **基石架构**   | **DIN**      | 引入Target Attention，解决定长向量瓶颈 | Local Activation Unit                 | 中短序列            | 否     |
|            | **BST**      | 将Transformer引入RecSys，捕获全局依赖 | Transformer Encoder                   | 中短序列            | 否     |
| **时序/结构**  | **DIEN**     | 模拟兴趣随时间的动态演化                | AUGRU (Attention Update Gate)         | 时序序列            | 否     |
|            | **DSIN**     | 发现并利用用户行为的Session结构         | Bi-LSTM (Intra) + Transformer (Inter) | Session序列       | 否     |
|            | **TiSSA**    | 解决绝对位置编码无法表达时间间隔的问题         | Time Interval Aware Attention         | 带时间戳序列          | 否     |
| **多维/上下文** | **DMT**      | 区分点击、加购、购买等不同行为的语义          | Multi-Transformer + MMoE              | **多行为序列**       | **是** |
|            | **DSTN**     | 引入页面展示的上下文广告(Context)       | Spatiotemporal Interaction Layer      | 历史+上下文          | 否     |
|            | **DHAN**     | 利用Item的层级结构(Cat/Brand)      | Hierarchical Attention                | 层级属性序列          | 否     |
|            | **HUP**      | 用户分层兴趣和用户行为有细粒度“微行为”        | Pyramid RNN (Behavior-LSTM Cell)      | 微行为 + 层级属性序列    | 否     |
| **长序列/工程** | **SDM**      | 解决长短期兴趣不一致及计算量问题            | LSTM (Short) + Attention (Long)       | 长+短混合           | 否     |
|            | **MIMN**     | **存算分离**，解决超长序列存储与延迟        | NTM (Memory Network) + UIC            | **超长序列(1000+)** | 否     |
|            | **TransAct** | 工业级大规模长短期混合建模               | Hybrid Transformer (Short+Sampled)    | 混合序列            | **是** |


## 参考
- https://zhuanlan.zhihu.com/p/521722722