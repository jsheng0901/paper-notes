# Deep Multifaceted Transformers for Multi-objective Ranking in Large-Scale E-commerce Recommender Systems

# 标题
- 参考论文：Deep Multifaceted Transformers for Multi-objective Ranking in Large-Scale E-commerce Recommender Systems
- 公司：JD
- 链接：https://guyulongcs.github.io/files/CIKM2020_DMT.pdf
- Code：https://github.com/guyulongcs/CIKM2020_DMT
- 时间：2020
- `泛读`

# 内容

## 摘要
- 问题：
  - 现有的推荐算法通常基于用户的历史点击序列，通过优化单个任务，来学习商品的排序得分，但很少同时建模用户的多种行为，或多种目标，例如点击率和转化率。
- 方法：
  - 建立多个Transformers同时建模用户多种行为序列，利用MMoE优化多个目标。并且加入bias net 缓解训练数据的选择偏差。
  - 本质上是多个用户行为序列 + 多任务的框架

## 1 介绍
- 三大问题：
  - 多目标优化：需要同时优化CTR、CVR等多个可能冲突的目标。
  - 行为复杂性：用户有多种行为（点击、加购、下单），它们含义不同，且对不同目标的重要性不同，但现有模型未能有效利用。
  - 数据偏差：存在位置偏差（物品因排位高而被点击）和邻近偏差（周围物品的竞争影响），影响模型学习的真实性。
- 方法：（DMT框架的三大组件，对应解决三个问题）
  - 多目标联合优化：采用MMoE结构来灵活学习不同任务（如CTR、CVR）与不同行为兴趣之间的关系，自动平衡多目标，解决任务冲突。
  - 多行为序列建模：使用多个独立的Transformer分别处理不同类型的用户行为序列（如点击序列、加购序列），从而提取用户多个维度的精细兴趣向量。
  - 显式偏差建模：引入一个偏置网络，专门学习并量化位置、邻近上下文等带来的偏差，从而在训练中“去偏”，让模型学习到用户真实的兴趣。
- **主要贡献**：
  - 通过建模用户多种类型的行为序列，研究了电子商务中的多目标推荐问题。
  - 提出了DMT，利用多个Transformer建模用户多样的行为序列，使用多门混合专家联合优化多目标，并采用一个偏置深度神经网络来减少电子商务推荐系统中的偏差。
  - 进行了广泛的实验，证明DMT在点击和转化任务上均大大优于最先进的基线方法。DMT已部署在京东的商业推荐系统中，每年为收入贡献数十亿美元。

## 2 RELATED WORK

### 2.1 CTR prediction
- 传统强基准：GBDT，因其综合优势在工业界仍被广泛使用。
- 深度学习范式：主流的 Embedding&MLP 范式。
- 研究演进方向：从简单聚合 -> 引入注意力（DIN） -> 捕捉时序演化（DIEN） -> 构建层次兴趣（HUP） -> self-attention
- 作者指出了当前研究的局限性：大多数模型只处理单一行为类型（如仅点击）和单一优化目标（如仅CTR）

### 2.2 Multi-task Learning for Recommendation
- 问题：
  - 特定场景模型：一些多任务模型针对性强，通用性不足。
  - Shared-bottom 架构：这是通用但基础的方法，存在硬参数共享的缺点，当多个任务目标差异较大或存在冲突时，共享底层参数会相互干扰，导致性能下降。
  - MMoE：通过引入多个“专家”和一个门控网络，软性地共享信息，能更好地处理任务间的关系与冲突，是当前的主流方案。
  - 其应用位置的局限：现有的MMoE通常作用在原始特征输入层或浅层的共享网络上，创新性有限。
- 创新点：
  - 将MMoE的应用位置提升了一个层次。不是直接处理原始特征，而是作用在已经由多个Transformer处理过的、高质量的多面性用户兴趣向量之上。
  - 本质上是将多个sequence的结果进行pooling，采用了门控机制控制信息共享和融合。

### 2.3 Unbiased Learning to Rank
- 问题：
  - 用于训练模型的数据（用户点击日志）存在选择偏差，其中位置偏差是最典型的例子（排在前面的物品更容易被点击，无论其实际相关性如何）。
  - 直接用有偏数据训练，模型学到的不是真实的“用户-物品”相关性，而是有偏日志中的模式，导致线上效果并非最优。
- 方法：
  - 传统方法：采用两阶段方式，需要训练一个独立的“倾向得分”模型来估计和纠正偏差（例如，一个物品因为其位置而被点击的概率）。
  - 新兴趋势：采用端到端方式，在一个统一的深度模型内部同时进行主任务预测和偏差建模。

## 3 PROBLEM FORMULATION
- CTR
- CVR = CTR + CR

## 4 METHOD
<p style="text-align: center">
    <img src="../../pics/DMT/DMT_4_模型结构.png">
      <figcaption style="text-align: center">
        DMT_模型结构
      </figcaption>
    </img>
  </p>

### 4.1 Input and Embedding Layers

#### 4.1.1 Categorical features
- 用户的多样行为序列
  - 每种类型的用户行为都表示为一个变长的物品序列。给定目标用户 *u*，我们模型的输入是目标物品和用户的多种类型行为序列 *S = ⟨x1, x2, ..., xT⟩*
  - 其中 T 是序列长度。第 *i* 个元素 xi = (ti, pi) 表示用户 *u* 在时间 ti 对物品 pi 执行了一个行为
  - 这里我们只考虑三种序列，点击序列 Sc、加入购物车序列 Sa 和订单序列 So
- Embedding Layer
  - 对于每个产品 pi，我们使用其产品ID、品类ID、品牌ID和店铺ID信息来表示它
  - 把这些ID， 𝑝𝑖 产品ID, 𝑐𝑖 品类ID, 𝑏𝑖 品牌ID, 𝑠𝑖 店铺ID，逐一转化成embedding，然后拼接起来变成一个 ei vector

#### 4.1.2 Dense features
- 总共615 个稠密特征，主要可分为三类：
  - 物品画像特征（如点击次数、CTR、CVR、评分）。
  - 用户画像特征（如购买力、偏好的品类和品牌）。
  - 用户-物品匹配特征（如物品是否match用户的性别或年龄）和用户-物品交互特征（如在时间窗口内对该物品所属品类的点击次数）。
- Z-score 标准化方法对稠密特征进行归一化

### 4.2 Deep Multifaceted Transformers Layer

#### 4.2.1 Deep Multifaceted Transformers
- 3个独立的 Deep Interest Transformers 层 (不同的参数) 去建模用户点击序列，加购序列和下单序列
  - 点击：量大、频繁，代表了短暂、浏览型的兴趣（短期）。
  - 加购：量少，代表了经过初步筛选、意向更强的兴趣（中期）。
  - 下单：量最少，代表了最强、最确定的兴趣和偏好（长期）。
- 本质上是模拟用户的不同时间段的行为
  - 点击序列：7天内最近点击的50个商品。
  - 加购：最近10个加入购物车的商品
  - 下单：一年内所有的下单

#### 4.2.2 Deep Interest Transformer
- encoder 模型学习 sequence 之间的关系
- decoder 模型学习用户关于目标商品的兴趣向量
- Self-attention blocks
  - Multi-head Self-attention Layers
  - Point-wise Feed-Forward Networks
  - 直接借鉴 NLP 里面的一模一样
- Positional encoding
  - Sinusoidal Positional Embedding (pos_sincos)，和NLP里面的思路一样
  - Learned Positional Embedding (pos_learn)，变成可学习的 embedding，然后和前面的 embedding 相加
  - 实验中显示可学习的 Positional Encoding 效果更好一些
- Encoder
  - 和 Transform 里面一样的用法，包含 Multi-head Attention 和 FFN，和上面的 Positional encoding
- Decoder
  - 和 Transform 里面有点不一样，decoder使用目标商品做为query，encoder的输出做为keys和values
  - decoder学习目标商品和历史序列中每个商品的attention score
  - 最终得到一个 interest vector 捕捉到用户的历史演化兴趣和目标商品的关系
  - 兴趣向量随着不同的目标商品而发生变化，这样可以提高了模型的表达能力
  - **本质上就是 seq - seq 的关系，解码的时候用 target item 来当第一个字符，下一个学习到的输出便是最终的 interest vector**

### 4.3 Multi-gate Mixture-of-Experts Layers
- 输入：
  - Deep Multifaceted Transformers Layer + dense layer （前面三个序列的输出 + 用户、商品、dense特征进行 + target item embedding）拼接起来
- 输出：
  - N 个 expert，k 个 task，k 个 gate，最终 k 个输出对应每个 task

### 4.4 Bias Deep Neural Network
- 问题：
  - 训练数据（用户点击记录）存在固有偏差，隐式反馈进行训练，因为数据本身就是之前的推荐系统输出的结果，不能反映用户的真实偏好。
  - 用户被“引导”了：用户只能看到系统展示的物品，他们的点击行为受展示位置和周围环境严重影响。
- 研究了两种偏差：
  - 位置偏差：指用户倾向于点击显示在列表更靠前位置的物品。每个物品的位置可以定义为屏幕中的索引号或页码。例如，排在第一位和第一页的物品更容易被点击。
  - 邻近偏差：指点击某个物品的概率可能受到其邻近产品的影响。（例如，一个商品被大量同质化商品包围时，点击率可能会被分散）。
- 方法：
  - 偏置深度神经网络
    - 目标：专门量化由位置和邻近环境带来的“虚假”点击概率。
    - 输入：使用可直接量化的偏差特征
      - 对于位置偏差，输入是目标物品的索引号或页码。
      - 对于邻近偏差，输入是目标物品及其最近K个相邻物品的品类。K=6 在实验中。
    - 将稀疏的偏差特征嵌入为低维向量，并将其输入到具有ReLU激活函数的多层感知机中。最后拼接到4.3的输出结果，直接到计算loss的部分
    - 给定偏差特征 xb，目标物品的选择偏差 yb 的计算公式为：yb = NNB(xb) 其中 NNB 就是偏置深度神经网络。
- **本质上是吧偏差信息喂给模型，然后让模型来学习，可以参考TiSSA里面的思路把空间信息也就是邻近偏差单独 sequence 建模。**

### 4.5 Model Training and Prediction

#### 4.5.1 Training
- 在训练阶段，对于每个任务k ，预估分数 yk 由从多任务学习层的 uk 和深度偏差网络 yb 使用sigmoid函数得到。𝑦𝑘 = 𝜎(𝑢𝑘 + 𝑦𝑏)
- 对于每个任务使用交叉熵损失函数，总的loss为每个目标的loss加权和
- 和大部分的多任务一样，只是每个任务都拼接了一个深度偏差网络 yb，并且如何设置权重对于每个任务是个问题

#### 4.5.2 Prediction
- 最终每个任务的预测分数为：𝑦ˆ𝑘 = 𝜎(𝑢𝑘)
- 然后最终预测结果为每个人的 𝑦ˆ𝑘 的加权平均，权重 𝑤k 由离线调参 grid search 和在线 A/B testing 选出


## 5 EXPERIMENTAL SETTINGS

### 5.1 Dataset
- 7 天 training
- 第 8 天 test

### 5.2 Baselines
- GBDT
- DNN
- DIN
- DIEN
- DMT𝑡𝑟𝑎𝑛𝑠：no Bias network and no MMoE

### 5.3 Evaluation Metrics
- task click and order
- offline:
  - AUC, RelaImpr, Precision@K and MRR@K
- online:
  - CTR, CVR and GMV

## 6 EXPERIMENTAL RESULTS

### 6.1 Comparison with Baselines
- 使用dense feature比不使用dense feature效果要好，所有模型的auc提升2个点。但是个人感觉，文章中说有dense特征迭代了5年，有200多个，一下子去掉200个有效特征AUC肯定会降低，这里的对比不太公平。
- 对比tran的效果，加入 Bias net 和 MMoE 的效果提升也很大。证明在京东推荐场景下，bias偏差影响很大。同时多个目标同时建模，MMoE也很关键。

### 6.2 Effectiveness of Components in DMT

#### 6.2.1 Deep Multifaceted Transformers
- pos_learn 效果最好，也就是可学习的position embedding
- click+cart 同时建模比 click 效果好，但是 click+cart+order 三个同时建模就差了。因为点击和下单相关性低。尤其是贵的商品，例如：电脑。对于便宜的快消品，点击和下单关系较大，例如牛奶。多个任务直接一起学习会导致这两种情况冲突。

#### 6.2.2 Multi-task Learning
- 使用MMoE能更进一步的提升效果，效果最好
- 可以把每个expert gating 之后的结果 plot出来对比，哪个expert在哪个task上面权重最高

#### 6.2.3 Bias Deep Neural Network
- 对于CTR这里加入了position index/page反而还下降了，加入neighbor 效果有提升。解释了相邻产品对于当前产品的点击的影响。
- 对于CVR这里加入了position index/page都有小幅度提升，neighbor 也不明显。解释了用户买不买东西和位置有一定的关系。

### 6.3 Online A/B Testing
- Compared with the GBDT, DMT improves the CTR, CVR and GMV by 18.8%, 19.2% and 17.9% respectively

## 7 CONCLUSION
- 模拟了用户的不同行为序列（点击序列、加购序列），从而提取用户多个维度的精细兴趣向量
- 用MMoE来融合不同行为行为序列，学习不同任务（如CTR、CVR）与不同行为兴趣之间的关系
- 加入了 Bias Net 来专门学习并量化位置、邻近上下文等带来的偏差


# 思考

## 本篇论文核心是讲了个啥东西
- 提出了用户不同行为序列可以有不同的兴趣代表，可以使用多个独立的Transformer来分别处理用户不同类型的行為序列（点击、加购、下单），以提取用户短期、中期、长期的多元化兴趣向量。
- 得到这些兴趣向量后，模型使用多门混合专家（MMoE） 结构来同时优化多个目标（如点击率CTR、转化率CVR），并能灵活处理不同任务间的关联与冲突。
- 模型还集成了一个偏置深度神经网络，专门用于建模和减少训练数据中的选择偏差（如位置偏差和邻近偏差），从而学习用户更真实的兴趣。
- **本质上是模拟了用户的不同时间段的多行为序列建模，MMoE的融合方式很有意思，并且证实了长期的purchase行为和当下的点击行为并没有太多关系，可以用门控机制来平衡**

## 是为啥会提出这么个东西，为了解决什么问题
- 问题：
  - 多目标排序的复杂性：
    - 电商业务不仅关心点击率（CTR），更关心转化率（CVR）、GMV等。这些目标可能相互关联也可能冲突，传统的单一模型或简单的共享底层网络难以有效平衡。
  - 用户行为信息的利用不充分：
    - 用户有多种行为（点击、加购、下单），它们代表了不同强度和阶段的兴趣。然而，大多数现有模型只关注单一的点击行为序列，浪费了其他宝贵的行为信号。
  - 训练数据中存在严重偏差：推荐系统的日志数据存在选择偏差，例如：
    - 位置偏差：用户更倾向于点击排在顶部的物品。
    - 邻近偏差：一个物品是否被点击受其周围竞争物品的影响。
  - 直接在这些有偏数据上训练模型，会导致模型学到有偏的模式，而非真实的用户兴趣，从而影响线上效果。
- 方法：
  - MMoE机制
  - 多行为序列建模
  - Bias Net 设计

## 为啥这个新东西会有效，有什么优势
- 对比DIN
  - 模拟了多行为，并且 transformer 结构，提取能力更强
- 对比DHAN，HUP 分层次模型
  - 模拟了多行为，并且加入了MMoE来融合，分层模型都没有考虑如何融合不同层次的输出
- 对比大部分序列模型
  - 提出了一个 Bias Net，虽然和 sequence 并没关系

## 与这个新东西类似的东西还有啥，相关的思路和模型
- 基于用户行为序列的模型：
  - DIN：
    - 相关思路：使用注意力机制，根据候选物品激活相关的历史行为。
    - 不同点：DIN没有使用Transformer，且通常处理单一类型的行为序列，未对行为类型和时间尺度进行区分。 
  - DIEN：
    - 相关思路：使用GRU序列模型来捕捉用户兴趣的演化。
    - 不同点：DIEN主要关注兴趣在时间上的变化，而DMT更关注不同行为类型所体现的兴趣侧面，并使用更强大的Transformer作为序列建模核心。
  - HUP：
    - 相关思路：强调了兴趣的层次性。
    - 不同点：HUP的层次体现在“物品-类别”的抽象层级上，而DMT的“多面性”主要体现在不同类型的行为序列上。并且HUP是单一sequence建模，DMT是混合sequence建模。

## 在工业上通常会怎么用，如何实际应用
- 特征工程上面，每个item采用了，从小到大的不同ID，（产品ID、品类ID、品牌ID和店铺ID）的处理方式可以借鉴。
- 三种行为的sequence的长度也可以参考如何选择，来满足长期，中期和短期，三种时间段。
- Transformer在每个sequence的使用方式可以考虑
  - encoder 模型学习 sequence 之间的关系
  - decoder 模型学习用户关于目标商品的兴趣向量
- MMoE 这个可以试一试对所有的需要融合不同 sequence 建模的结果的模型，本质上是一种pooling做法
- Bias Net 可以考虑特别是上下文的偏差，全新的方向其实和 sequence 关系不大。这里还可以参考 DSTN 里面的空间辅助信息思路一样。


