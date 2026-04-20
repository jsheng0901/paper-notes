# SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

# 标题
- 参考论文：SDM: Sequential Deep Matching Model for Online Large-scale Recommender System
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/1909.00385
- Code：
  - https://github.com/alicogintel/SDM
  - https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/models/sdm.py
- 时间：2019
- `泛读`

# 内容

## 摘要
- 问题：
  - 传统CF方法无法有效捕捉用户动态变化的兴趣。
  - 现有序列模型的两个固有缺陷：
    - 用户一次会话中可能包含多种不同的兴趣（例如，同时浏览手机和衣服）。
    - 用户的长期历史行为复杂多样，如何从中筛选出于当前短期兴趣相关的部分并进行融合，是一个挑战。
- 方法：
  - 多头注意力：同时捕捉会话中并存的多种兴趣
  - 长短兴趣门控融合：用于自适应的融合长期和短期偏好

## 1 INTRODUCTION
- 问题：
  - 目前基于 item-based CF 模型对静态的 user-item 交互进行建模，无法很好地捕获用户整个行为序列中的动态变换dynamic transformation。因此，这类方法通常会导致同质推荐。为了准确了解用户的兴趣interest 和偏好preference，应该将序列的顺序信息sequential order information 融入召回模块。
- 方法：
  - 最近的交互 session 称为短期行为 short-term behaviors
  - 历史的其它session 称为长期行为long-term behaviors
  - 我们对这两部分分别建模，编码短期行为和长期行为的固有信息，这些信息可用于表示用户的不同兴趣level
- 短期偏好
  - 问题：
    - RNN表现的很不错，但是可能受到由于用户的随机行为引起的兴趣转移影响
    - 同时用户会有多个兴趣点，如类目category、品牌brand、颜色、款式style 和店铺声誉reputation 等等。在对最喜欢的 item 做出最终决定之前，用户会反复比较很多 item
  - 方法：
    - multi-head structure 通过代表不同视角views 的偏好来自然地解决多重兴趣 multiple interests 的问题 
    - 同时这种 self-attention，可以通过过滤掉偶然发生的点击来表达准确的用户偏好，也就是不那么受到兴趣转移的影响
- 长期偏好
  - 问题：
    - 用户长期行为是复杂和多样的，与当前短期购物先关的信息可能占据很少的一部分，也就是当前短期session 相关的长期用户偏好无法在整体长期行为representation 中突出显示
    - 需要考虑如何保留长期行为representation 中与短期session 相关的信息
  - 方法：
    - 门控融合模块gated fusion module，合并全局（长期）和局部（短期）偏好特征。
    - 输入是用户画像 embedding、长期representation 向量、短期 representation 向量。然后学习门控向量gate vector 来控制 LSTM 中不同gate 的融合行为fusion behaviors，以便模型可以精确地捕获兴趣相关性以及用户对长期兴趣/短期兴趣的attention
- **主要贡献**：
  - 通过考察短期 short-term 行为和长期 long-term 行为，提出了 SDM 模型。对短期行为和长期行为分别建模，代表不同 level 的用户兴趣。
  - 提出通过 multi-head self-attention 模块对短期session 行为进行建模，从而编码和捕获多种兴趣倾向。
  - 提出门控融合模块用于有效结合长期偏好和当前购物需求，从而结合它们的相关信息而不是简单的组合。
  - SDM 模型已经在线上有效运行并取得了显著的提升

## 3 THE PROPOSED APPROACH

### 3.1 Problem Formulation
- session 的生成规则：
  - 后端系统记录的、具有相同 session ID 的交互属于同一个 session。
  - 时间间隔小于 10 分钟（或者更长，这取决于具体场景）的相邻交互也合并为一个 session。
  - session 的最大长度设置为 50，即 session 长度超过 50 时将开始新的 session。
- 用户的 latest session 为最近的那个 session，被视为短期行为 Su
- 长期行为 long-term behaviors 是过去 7 天的，发生在 Su 之前的行为记作 Lu

### 3.2 Training and Online Serving
- 在训练过程中
  - 时刻 t 的positive label 是下一个交互 item
  - negative labels 采用 log-uniform sampler
  - 然后由 softmax layer 生成预测类别概率。这称作 sampled-softmax。
  - 最后采用 CE Loss
- online serving
  - 用户的历史行为（短期行为 Su 和长期行为 Lu ）被输入到推断系统中，生成用户行为向量 O_t
  - KNN 搜索系统根据内积检索最相似的 item，然后推荐 top-N 个 item

### 3.3 Input Embedding with Side Information
- 用户不仅关注特定item 本身，还关注品牌、店铺、价格等，所以只用item ID 的特征级别feature level 对 item 进行编码远远不能令人满意。
- 用item ID、叶子类目、一级类目、品牌、店铺 不同的特征尺度feature scales 描述一个 item
- 每个输入item，把所有这些ID 做 embedding 然后拼接起来
- 同样对于用户 也可以用不同的feature 进行拼接

<p style="text-align: center">
    <img src="../../pics/SDM/SDM_3.2_模型结构.png">
      <figcaption style="text-align: center">
        SDM_模型结构
      </figcaption>
    </img>
  </p>

### 3.4 Recurrent Layer
- 给定用户 u 的 embedding 短期行为序列
- 用 LSTM 网络作为循环单元 recurrent cell 在每一个时刻 t
- 最后得到每个时刻 t 的 hidden output vector h_t
- 学习到的称之为序列偏好 representation
- 本质上和其它的短期模型一样，先用LSTM学习一个序列关系

### 3.5 Attention Mechanism
- 用户通常会交替浏览一些不相关的 item，这种情况称作偶然点击causal clicks。不相关的动作会以某种方式影响序列中 h_t 的representation。利用多头自注意机制，把这种无关动作减少
- 输入是上层 LSTM 每个时刻的 h_t。

#### 3.5.1 Multi-head Self-Attention
- 直接照搬 Transformer 里面的多头自注意力机制
- 多头可以学习到不同维度的兴趣点

#### 3.5.2 User Attention
- 相似历史行为但是不同用户，喜欢的偏好也有点不一样。
- 用 user 的 e_u 作为 query，和上面多头的输出 x_u，每一个时刻计算 attention score，再最后对每个时刻加权一下，得到最终短期行为 s_t_u

### 3.6 Long-term Behaviors Fusion
- 用户一般会在各个维度上积累了不同层次level 的兴趣。可能经常访问一组相似的店铺并重复购买属于同一个类目的item 。
- 还是需要从不同的特征尺度 feature scales 对长期行为 Lu 进行编码
- Lu = {Luf |f ∈ F } 包含多个子集:
  - Lu id (item ID)
  - Lu leaf (leaf category)
  - Lu cate (first level category)
  - Lu shop (shop)，比如这个Lu shop，包含用户 u 过去一周内互动过的店铺
  - Lu br (brand) 
- 上述每个子集转化成 embedding 后，这里的ID embedding matrix 和 3.3 里面的 W 一样 share。
- 然后使用用户画像 embedding e_u 作为 query 向量来计算 attention score，和 3.5.2 里面的最后一步一样。只是输入变成了 embedding 的结果。
- 加权得到的 z_u，最后拼接起来进入 MLP 层 p_u = tanh(W_p * z_u + b)。最终得到用户的长期行为 representation p_t_u。
- 区别：
  - 短期行为使用序列建模，长期行为使用 attention 建模。二者使用相同的特征转换矩阵。
  - 但是短期行为建模使用不同特征尺度 embedding 的拼接作为 item 的 representation，而长期行为建模直接对每个特征尺度的 embedding 来进行。
- **本质上就是计算长期下和不同层次的行为，计算用户的相似度。和多序列建模的思路一样，只是这里是item的不同维度的信息，不是行为。**
- 最后为了结合长期行为和短期行为，我们精心设计了一个以 e_u，s_t_u，p_u 作为输入的门控神经网络gated neural network。
- 门向量 G_u_t 决定 t 时刻这三个向量的融合比例，G_u_t = sigmoid(W1 * eu + W2 * s_u_t + W3 * p_u + b)
- 最终的用户行为向量为 o_u_t = (1 − G_u_t) ⊙ p_u + G_u_t ⊙ s_u
- **这里本质上和LSTM里面的门控机制是一样的逻辑，对每个向量的每个维度都赋予一个参数来控制融合的比例，然后根据比例再对长期和短期进行加权融合。这个思路可以精细到每个维度的融合，而这里每个维度可能就是用户兴趣特征的不同层子，比如item，品牌，类型。保证了融合的时候不再是整体所有维度的统一融合，而是自适应的学习需要哪个维度多一些，进行最终融合。这个地方提供了一种多个特征融合的思路**

## 4 EXPERIMENT SETUP
- 0.001 Adam
- batch size = 256
- embedding 和模型参数都是从头开始学习的，没有预训练
- dropout rate = 0.2 的多层 LSTM，并且在 LSTM 和残差链接中间
- LSTM 的 hidden size 在离线和在线实验中分别设置为 64 和 128
- multi-head attention 结构，在离线和在线实验中分别将 head 数量设置为 4 和 8
- 加入 layer normalization 和 residual adding
- item embedding 向量维度、短期行为向量维度、长期行为向量维度、用户行为向量维度和 LSTM 保持一致。也就是 64 或者 128

## 5 EMPIRICAL ANALYSIS
- 最终结果，加入长期偏好，加入门机制，加入辅助信息（shop ID 之类的），明显提升模型效果
- head 的数量 从 1 - 4 开始越来越好，之后越来越差
- 门口机制，对比简单的加，点成，拼接，明显效果更好

## 6 CONCLUSIONS
- 提出一种长期和短期兴趣的融合模型
- 提出多头自注意力机制来学习短期不同层次的兴趣
- 提出 gate 融合机制，来融合短期和长期兴趣


# 思考

## 本篇论文核心是讲了个啥东西
- 提出了长期和短期的融合模型
- 对于短期：
  - 多头自注意力模块：
    - 能够识别出单个会话内用户可能存在的多个并行兴趣（例如，在一次浏览中同时看手机和裙子）。
  - User attention net:
    - 寻求和用户的交互，能更精准的学习同样行为不同用户的偏好
- 对于长期：
  - 长短时门控融合模块：
    - 设计了一个门控机制，像智能过滤器一样，从复杂多样的长期行为中，自适应地筛选、融合与当前短期会话最相关的部分，忽略不相关的部分。
  - User attention net:
    - 学习长期兴趣中不同类型层次的兴趣和用户画像的关系
- **本质上和DMT一样，同样是模拟了多序列建模，这里短期考虑单一序列，长期考虑多序列，同时提出了一种 gate 机制来融合长短期。**

## 是为啥会提出这么个东西，为了解决什么问题
- 问题：
  - 传统方法的局限：
    - 业界常用的基于物品的协同过滤等方法，本质上是静态的，无法捕捉用户兴趣的动态变化和演化过程。
  - 现有序列模型的不足：
    - 短期偏好问题：
      - 许多序列模型（如RNN）将整个用户会话编码为一个单一的向量，这会导致会话中多个不同兴趣被模糊地混合在一起，降低了表征的清晰度和匹配精度。
      - 同时RNN模型和容易受到兴趣转移的影响，导致遗忘了之前有用的兴趣。
    - 长短兴趣融合问题：
      - 当尝试结合长期行为时，现有方法往往简单拼接或用注意力加权，未能有效处理长期行为的复杂性和冗余性。长期行为中可能包含大量与当前场景无关的兴趣，直接全盘融合会引入噪音。
- 方法：
  - 数据上提出不仅仅使用item ID，同时使用category ID，shop ID 之类的来表达不同层次的兴趣
  - 多头自注意力机制来学习短期的不同兴趣
  - gate 机制来自适应的学习兴趣融合

## 为啥这个新东西会有效，有什么优势
- 对比大部分短期序列模型
  - 加入了长期兴趣的考虑
  - 并且考虑了长期兴趣的多个层次方向（item，category，shop，brand 之类）
  - 同时提出了一个 融合 gate，来自适应学习，减少长期兴趣的noise

## 与这个新东西类似的东西还有啥，相关的思路和模型
- 基于用户行为序列的模型：
  - 这里主要SDM只用于召回阶段，计算的也是item 和 user 的相似度。

## 在工业上通常会怎么用，如何实际应用
- 短期行为中，考虑增加不同层次的兴趣ID
- 考虑短期行为中的多头自注意力机制，query 可能考虑用 candidate item 而不是 user
- 长期行为中考虑多序列的融合 attention net 的机制来学习和 candidate item 的关系
- 两种序列融合 gate 值得尝试，可以对比一下 MMoE 的效果。

## 参考
- https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/chapters/13_dnn_rec_system2.html
- https://zhuanlan.zhihu.com/p/141411747