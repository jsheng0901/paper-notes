# TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest

# 标题
- 参考论文：TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest
- 公司：Pinterest
- 链接：https://arxiv.org/pdf/2306.00248
- Code：
  - https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/TransAct/src/TransAct.py
  - https://github.com/pinterest/transformer_user_action
- 时间：2023
- `精读`

# 内容

## 摘要
- 问题：
  - 传统的序列推荐方法要么对实时用户行为进行端到端学习，要么以离线批处理方式单独学习用户表示
  - 也就是要么从零开始训练embedding，要么使用离线预训练的embedding
- 方法：
  - 提出了TransAct，一个从用户实时活动中提取其短期偏好的序列模型。
  - 混合排序方法
    - TransAct（端到端序列模型）：
      - 专门处理实时用户活动，快速捕捉用户即时、短期的兴趣变化，保证推荐的新鲜度和响应速度。
    - 批量用户嵌入（离线模型）：
      - 基于长期历史数据（如过去几周）训练，刻画用户稳定、长期的兴趣画像，计算成本低且表征丰富。
  - **本质上就是把短期实时和长期稳定，进行了融合，还是长短期用户兴趣的思路，重点是怎么学习短期和长期，并且怎么融合。同时保证工业级速度。**
  - 该模型已经成功上线，提出了一个工业级混合推荐系统架构设计思想

## 1 INTRODUCTION
- 问题：
  - 一些模型要么只使用实时用户行为，要么只使用从长期用户行为历史中学到的批量用户表征。并没有同时考虑如何融合。
  - 如何有效利用超长用户行为序列（包含实时信号和长期历史）同时满足在线服务的低延迟要求。
- 方案：
  - 混合架构：
    - 实时流近期：使用TransAct（轻量级Transformer）处理近期短序列，捕捉即时兴趣变化，保证响应速度。
    - 批量流长期：使用离线预计算的用户嵌入（如通过其他模型从长期历史生成），承载丰富、稳定的长期兴趣画像，成本低廉。
    - 最后是把上述两种进行混合
- **主要贡献**：
  - 描述了 Pinnability，即 Pinterest 首页信息流生产排序系统的架构。该个性化推荐产品占据了Pinterest用户整体互动的大部分。
  - 提出了 TransAct，这是一个基于 Transformer 的实时用户行为序列模型，能有效从用户近期行为中捕捉其短期兴趣。我们证明，将TransAct与每日生成的用户表征结合为混合模型能在 Pinnability 中取得最佳性能。这一设计选择通过全面的消融研究得到了验证。
  - 描述了在 Pinnability 中实现的服务优化，它使得在模型中引入 TransAct 所带来的65倍计算复杂度增长变得可行。具体而言，我们通过优化实现了之前基于CPU的模型能够进行GPU服务。
  - 描述了使用 TransAct 在真实推荐系统上进行的在线A/B实验。我们展示了线上环境中的一些实际问题，如推荐多样性下降和互动衰减，并提出了解决方案。

## 2 RELATED WORK

### 2.1 Recommender System
- 传统方法：
  - 协同过滤：基于用户或物品的相似性，核心问题是数据稀疏和冷启动。
  - 因子分解机：作为CF的扩展，能更好地处理稀疏数据，学习特征的隐向量。
- 深度学习模型：
  - 核心范式：Embedding + MLP（先嵌入，后通过神经网络学习）。
  - 模型重点：从 Wide & Deep（记忆与泛化结合） 到 DeepFM、DCN（自动、高效地学习特征交叉），演进的核心是如何更好地建模特征间的复杂交互。
- 关键问题：
  - 静态性：这些模型主要依赖用户和物品的静态画像特征。
  - 非序列性：它们将用户的历史行为视为一个无序的集合，完全忽略了行为之间的顺序和时间动态。
  - 结论：无法有效捕捉用户短期、动态变化的兴趣

### 2.2 Sequential Recommendation
- 技术演进脉络：
  - 早期（传统ML）：马尔可夫链、会话KNN。局限：难以捕获长期模式。
  - 中期（深度学习）：
    - RNN：成为主流，擅长序列建模。
    - CNN：擅长捕捉局部依赖。
    - 当前（注意力时代）：
      - 注意力/自注意力：成为强大工具，能灵活衡量序列不同部分的重要性。
- 对比阿里的BST论文思路：
  - 位置信息可能并不关键：与BST等强调行为顺序的结论不同，在CTR预测场景中，位置编码带来的增益有限。
  - 更有效的设计：提出更好的早期特征融合和对行为类型进行嵌入可能是更实际有效的改进方向。

## 3 METHODOLOGY
- 先介绍 Pinterest 的主页排序模型 Pinnability
- 在介绍 TrancAct 如何学习用户行为序列特征

### 3.1 Preliminary: Homefeed Ranking Model
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_3.1_Pinnability结构.png">
      <figcaption style="text-align: center">
        Pinnability 结构
      </figcaption>
    </img>
  </p>

- 任务定义与架构：
  - 任务：point-wise 多任务预测，同时预测用户对单个候选物品（Pin）可能采取的多种动作（如点击、转存、隐藏）的概率。每个任务都包含正负样本。
  - 基础架构：基于经典的 Wide & Deep 范式。如图应该是stack结构而不是parallel结构。
  - 特征处理：综合运用 embedding（类别特征）、归一化（数值特征）和全秩DCN V2（进行显式、高阶特征交叉）。
  - 本质上就是把特征进行拼接（类别特征 + 数值特征 + 短期用户行为特征 （TransAct 提取） + 长期用户行为特征（离线预计算的用户嵌入））后送进DCN V2 stack结构做特征交叉，然后送入多任务每个head。
- 混合模型核心：
  - 实时流：TransAct（基于Transformer）处理近期行为，捕捉即时兴趣。
  - 批量流：PinnerFormer（一种批量用户嵌入模型）编码长期历史，表征稳定偏好。
  - 两路特征在模型中融合，构成了其实时-批量混合的本质。这里的融合方式看起来就是直接concat。
- 损失函数设计：
  - 双重加权机制：
    - 任务间权重：引入标签权重矩阵 𝑴，这是一个非对角的经验矩阵，用于精确刻画不同预测任务之间的关联与相互影响（例如，“转存”动作与“点击”动作的损失计算可以相互加权）。这里设置方式应该是人工有选择的经验方式。
    - 样本间权重：引入用户相关权重 𝑤𝑢，根据用户的人口统计属性（状态、性别、位置）对样本进行加权，使模型训练能够服务于特定的业务目标（如提升某地区或某用户群体的体验）。

### 3.2 Realtime User Action Sequence Features
- 实时用户行为提取问题：
  - 理想：序列越长，用户兴趣表征越丰富、越准确。
  - 现实：序列过长会导致特征获取和模型推理的延迟与成本急剧上升，影响线上服务。
- 解决方案：
  - 固定长度截断：统一截取最近100个行为作为一个合理的平衡点。
  - 标准化处理：对于不足100的行为进行 padding 0 填充，确保所有输入维度一致，便于批量计算。
  - 时间顺序：按时间倒序排列，确保模型最先看到最新的行为。
- 特征设计：
  - 每个行为包含三要素：动作类型（点击（click）、repin（收藏）和 hide（隐藏，属于负反馈））、物品嵌入（这里是 PinSage 嵌入，代表了被交互物品的内容语义，这里采用32维度的 pretrained embedding）。
  - **本质上就是表达了“用户在何时、以何种方式、与何种内容”进行了交互的关键信息。**

### 3.3 Our Approach: TransAct
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_3.3_模型结构.png">
      <figcaption style="text-align: center">
        TransAct 模型结构
      </figcaption>
    </img>
  </p>

- TransAct 用于提取用户历史行为中的兴趣 pattern，计算用户和候选 item 的相关分数

#### 3.3.1 Feature encoding
- 核心：
  - 用户的不同行为类型（如“转存”、“点击”、“隐藏”）蕴含了关于用户对物品兴趣强度和偏好方向的关键信号。将这些类型信息进行嵌入，能使模型理解不同行为的语义差异。
- 方法：
  - 行为类型嵌入：使用可训练的embedding table，将行为等类别映射为稠密向量，以编码行为的语义。
  - 物品内容嵌入：使用预训练的PinSage embedding，来表征物品本身的内容语义。
  - 特征拼接：将每个user sequence action 的这两个向量在特征维度上进行拼接，从而形成一个同时包含“用户做了什么”和“对什么做的” 的复合表征。

#### 3.3.2 Early fusion
- 核心：
  - 推荐算法中的early fusion指的是在模型的早期阶段，融合用户和候选的特征。比如计算候选和用户行为序列中元素的相关性
  - 可以更早期的建立用户历史行为和候选 item 之间的关系，更早的融合显性学习特征交叉
- 方法对比与选择：
  - 拼接法（Append）：即在用户行为序列的最后添加候选的 PinSage embedding。即TransAct的输入的序列长度为 100+1。由于候选 embedding 还没有发生任何动作，所以使用了全零向量作为占位向量。（这里理解的候选 item 应该是 trainable 的 embedding 向量）。此方法类似BST paper 里面的操作，把历史行为和候选item直接拼成一个新的序列。
  - 连接法（Concat）：将候选物品信息与序列中每一个历史行为增加一个候选的 PinSage embedding。即序列每一个元素的输入由三部分构成：action type embedding + history_i PinSage embedding + candidate PinSage embedding（这里貌似还是使用的是 pretrained embedding），这使得模型能够逐个、显式地计算候选物品与每个历史物品的相关性，更细粒度、更彻底的交互方式。
  - 结论：离线实验表明，连接法性能更优，因此被采用。

#### 3.3.3 Sequence Aggregation Model
- 核心：
  - 将 3.3.2 融合后的用户行为序列矩阵 U 聚合为一个能够代表用户短期偏好的表征向量。
- 架构选型：
  - 过程：对CNN、RNN、Transformer等流行架构的实验比较。
  - 决定：最终选择了Transformer编码器作为序列聚合器，这符合当前序列建模的主流趋势。
- 参数：
  - 轻量级设计：使用了相对简单的配置（2层，1头）。
  - 关键结论：位置编码在本任务中被证明无效，因此被省略。这是一个重要的实践发现，它表明在该推荐场景下，用户近期行为的绝对顺序或相对位置可能不如行为本身的内容（物品、动作类型）及与候选物品的交互重要。
  - **本质上和BST的思路一样，区别在于 Concat方式，可以更好的学习候选 item 和 历史行为 item 之间的关系。位置编码不重要说明在推荐系统中，历史行为的绝对顺序并没有太多意义，并且并且兴趣演化过程并不重要，这里和DIEN的思路有点不一样，值得思考业务下不同场景的需求不一样。**

#### 3.3.4 Random Time Window Mask
- 问题：
  - “兔子洞效应”，即模型过度适应用户近期（如几小时内）的密集、同质化行为，导致推荐结果越来越窄，损害用户体验的多样性和长期留存。
- 方法：
  - 核心思想：在训练阶段，随机地“隐藏”用户最近一段时间（从0到24小时中随机选择）的所有行为，强迫模型从更早、更多样化的历史行为中去学习用户兴趣。
  - 技术实现：在Transformer的自注意力机制前应用一个动态的时间掩码。也就是把 (𝑡𝑟𝑒𝑞𝑢𝑒𝑠𝑡 − 𝑇 , 𝑡𝑟𝑒𝑞𝑢𝑒𝑠𝑡) 这个时间段的数据 mask 掉。𝑡𝑟𝑒𝑞𝑢𝑒𝑠𝑡 是受到 request的时间，𝑇 是0到24小时中随机选择。
- 核心：
  - 仅训练，不推理：在训练时，模型被锻炼得不依赖近期窄兴趣，从而学到更稳健、更多元的长期兴趣表征。在线上推理时，移除掩码，模型可以充分利用包含近期行为的完整序列进行精准的实时预测，二者毫不冲突。
  - 随机性：时间窗口T是随机采样的，这增加了训练数据的扰动，提升了模型的鲁棒性。
  - 本质上就是一个mask，类似 BERT 里的随机 mask掉 15% 的输入数据。

#### 3.3.5 Transformer Output Compression
- 输出聚合策略（双路信息）：
  - 近期兴趣（头部列）：直接取 Transformer 输出的前 K 列。这基于一个合理的假设：在按时间倒序排列的序列中，序列前部（最近的几个行为）的编码结果最能反映用户的即时兴趣。
  - 长期偏好（池化摘要）：对剩下的 |S - K| 个输出进行最大池化，得到一个能概括序列全局信息的向量，它代表了用户在整段时间内的综合、稳定的兴趣倾向。
  - 这里一般设置超参数 K = 10
- 最终融合方式：
  - 特征拼接与展平：将代表不同时间尺度的两部分信息（近期K列 + 长期池化向量）拼接并展平，形成一个统一的用户短期兴趣表征向量 z。z ∈ R(𝐾+1) * d。
  - 最后送进 DCN V2 计算特征交叉

### 3.4 Model Productionization

#### 3.4.1 Model Retraining
- 问题：
  - 用户兴趣和行为是动态变化的。使用旧数据训练的静态模型会迅速“过时”，无法捕捉最新趋势，导致推荐效果衰减。
- 方法：
  - Pinterest 采用了每周重训练两次的节奏。
  - 目的：保证模型效果（维持用户参与率） 与控制计算成本之间找到的一个经过验证的最佳平衡点。

#### 3.4.2 GPU serving
- 问题：
  - 模型创新的效果（TransAct）被指数级增长的计算成本所威胁 （复杂度激增 65倍），传统的CPU推理已无法满足要求。
- 方案：
  - 从 CPU推理 转向 GPU推理，利用GPU的大规模并行计算能力来消化增加的复杂度，实现成本与延迟与之前的线上模型持平。
  - GPU inference 问题：
    - CUDA内核启动开销过大，具体分为：
      - Pinnability 乃至推荐模型通常处理数百个特征，这意味着存在大量的CUDA内核
      - 在线服务时的 batch size 很小，因此每个CUDA内核只需要很少的计算。面对大量的小型CUDA内核，启动开销远比实际计算本身昂贵。
  - 方法：
    - Fuse CUDA kernels 融合CUDA内核：
      - 问题：
        - 尽可能多地融合操作，因为发现像nvFuser这样的标准深度学习编译器，有许多剩余操作，通常需要人工干预。
        - 比如： embedding table lookup 模块，它包含两个计算步骤：原始ID到表索引的查找，以及表索引到嵌入向量的查找。由于特征数量庞大，这个过程要重复数百次
      - 实现：
        - 利用cuCollections来支持 GPU 上原始ID的哈希表，并实现了一个定制的统一embedding lookup 模块，将多个特征的查找合并为一次查找，从而显著减少了操作数量
        - 最终与稀疏特征相关的数百次操作减少为一次。
        - **本质上就是把多个特征做 embedding lookup 的次数进行融合为一次，并且实现 GPU 上面快速查找，这样可以降低开CUDA内核的个数**
    - Combine memory copies 合并内存：
      - 问题：
        - 上百个特征 Tensor 作为独立 Tensor 从 CPU 复制到 GPU 非常耗时
      - 实现：
        - 在复制前，把上百个特征 Tensor 合并成一个 Tensor buffer
        - 显著降低了复制的 scheduling 时间
    - Form larger batches 形成更大的batch size：
      - 问题：
        - 对于基于CPU的推理，更小的 batch size 更受青睐，以增加并行性并降低延迟。然而，对于基于GPU的推理，更大的 batch size 效率更高
      - 实现：
        - 之前，使用分散-聚合架构将请求拆分成小批量，并在多个叶子节点上并行运行以获得更好的延迟，换成直接使用原始请求中更大的批次在 GPU 上。
        - 为了补偿 cache 容量的损失，我们实现了一个同时使用DRAM和SSD的混合 cache。
        - **本质上这里同样可以降低开CUDA内核的个数**
    - Utilize CUDA graphs 利用CUDA图：
      - 问题：
        - 解决内核启动的固定开销
      - 实现：
        - 依靠 CUDA图 来完全消除剩余的小型操作开销。CUDA图将模型推理过程捕获为一个静态的操作图，而不是单独调度的操作，从而允许计算作为一个单一单元执行，没有任何内核启动开销。
        - **本质上是降低了 CUDA 的启动开销**

#### 3.4.3 Realtime Feature Processing
- 数据流转架构：
  - 源头：用户在前端的实时行为事件。
  - 消息队列：通过 Kafka 进行高吞吐、低延迟的数据流传输。
  - 实时处理：由 Flink 作业消费流数据，完成数据清洗（验证、去重、时间对齐）。
  - 特征存储：处理后的特征被持久化到 Rockstore（一种特征存储系统）中。
- 线上服务流程：
  - 按需触发：当线上服务收到推荐请求时，特征处理器被动态触发，从特征存储中提取所需的用户行为序列特征。
  - 格式转换：将存储的原始特征转换为排序模型（如 Pinnability）可以直接使用的 Tensor。

## 4 EXPERIMENT
- 线上和线下实验对比，使用 Pinterest 内部自己的数据

### 4.1 Experiment Setup

#### 4.1.1 Dataset
- 2 周的数据 train，最后一周的数据 test
- sampling 数据取决于，不同label的 statistical distribution 和 importance
- down sampling on negative label，因为是 point-wise，负样本太多了
- 3 B train instance，177 M user 和 720 M pins

#### 4.1.2 Hyperparameters
- realtime short term sequence length = 100
- action embedding dim = 32
- TransAct
  - 2 layers
  - dropout = 0.1
  - FFN hidden dim = 32
  - position encoding no use
- Adam
- batch size = 12000

### 4.2 Offline Experiment

#### 4.2.1 Metrics
- HIT@3
- 对每个 head 都会计算，并且对于 positive head 越高越好，negative head （hide）越低越好

#### 4.2.2 Results
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_4.2_模型结果.png">
      <figcaption style="text-align: center">
        TransAct 模型结果
      </figcaption>
    </img>
  </p>

- WDL: averaging pooling on user sequence
- BST: 两个版本，明显无法分辨 negative action
- TransAct 最好

### 4.3 Ablation Study

#### 4.3.1 Hybrid ranking model
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_4.3_消融实验结果.png">
      <figcaption style="text-align: center">
        模型结构消融实验结果
      </figcaption>
    </img>
  </p>

- 去掉短期 realtime TransAct 影响最大

#### 4.3.2 Base sequence encoder architecture
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_4.3_序列架构消融实验结果.png">
      <figcaption style="text-align: center">
        序列架构消融实验结果
      </figcaption>
    </img>
  </p>

- 简单的 average pooling 都可以提高效果
- 复杂的模型 CNN，RNN，LSTM 反而降低了效果

#### 4.3.3 Early fusion and sequence length selection
- concat 对比 append 效果最优
- sequence length 长度越长越好

#### 4.3.4 Transformer hyperparameters
- 4 layers, 384 FNN hidden dim 效果最好，但是 latency +30%，不能接受
- 最终选择 2 layers, 64 FNN hidden dim，latency 几乎没有增加

#### 4.3.5 Transformer output compression
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_4.3.5_模型输出压缩对比.png">
      <figcaption style="text-align: center">
        模型输出压缩对比
      </figcaption>
    </img>
  </p>

- 压缩 TransAct 的输出维度，保证 DCN V2 的输入大小
- 前 k 个 全部提取，后面的 max pooling，k = 10。基本上取的是最近的10次行为，也是用户 realtime 的最近的行为

### 4.4 Online Experiment
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_4.4_线上实验对比.png">
      <figcaption style="text-align: center">
        线上实验对比
      </figcaption>
    </img>
  </p>

- 基本在全局用户和Non-core用户中均取得核心指标得收益
- Non-core用户的定义是在过去的28天中没有pin过任何图片的用户，也就是不活跃用户的推荐效果也很好

## 5 DISCUSSION

### 5.1 Feedback Loop
- 发现：
  - 模型的线上价值（尤其是指标提升）在于启动一个自我强化的正向循环。
- 运作机制：
  - 模型提升体验：更好的模型（TransAct） → 提供更准、更快的推荐。
  - 用户行为改变：更好的体验 → 用户更愿意互动（点击、转存等），产生更高质、更丰富的行为数据。
  - 数据反哺模型：新的高质量数据 → 用于重训练 → 产生更强大的下一个迭代模型。
- 结果：
  - 形成一个不断增强的增长飞轮，带来持续提升的用户参与度。

### 5.2 TransAct in Other Tasks
<p style="text-align: center">
    <img src="../../pics/TransAct/TransAct_5.2_在其他任务上面的应用.png">
      <figcaption style="text-align: center">
        在其他任务上面的应用
      </figcaption>
    </img>
  </p>

- 明显 search ranking 上面也有提升

## 6 CONCLUSIONS
- 核心贡献：
  - 提出了 TransAct —— 一个利用Transformer架构处理实时用户行为序列以捕捉短期兴趣的模型。
- 核心架构与价值：
  - 创造性地采用了 实时与批量相结合的混合排序架构，兼顾了即时反馈和长期兴趣，并已成功在Pinterest的核心场景（首页信息流） 中实际部署。

  
# 思考

## 本篇论文核心是讲了个啥东西
- 同样是提出了长期和短期的融合模型
共同输入到一个基于 Wide & Deep 和 DCN V2 的主排序模型（Pinnability）中。

核心价值：这是一篇系统论文，不仅提出了新模型，更详细阐述了如何将计算复杂度激增（65倍）的先进模型，通过一系列工程优化（如GPU推理、内核融合）成功部署到生产环境，并解决了线上遇到的实际问题（如多样性下降）。
- 对于短期：
  - TransAct 一个基于 Transformer编码器 的模型，专门用于处理用户的实时行为序列（如最近100次交互），以捕捉用户短期的、动态变化的兴趣。
  - 其中包括：
    - 采用了混合行为序列，包含多种action 类型，同时有positive 和 negative 行为
    - 早期融合，采用 concat 形式融合用户行为和候选 item
    - 随机 time window mask，保证了多样性，不受到近期行为主导影响
- 对于长期：
  - 离线预计算的批量用户嵌入（代表长期、稳定的兴趣）相结合
  - 直接使用 pretrained 好的 embedding，降低了模型压力，同时可以利用更长的 sequence
- 最后压缩输出，保证了近期的全部信息，同时对长期行为进行pooling
- 提出了一系列工程上的优化，加速了实时行为序列学习
- **本质上和SDM一样，长短期都考虑，这里短期是单一序列，长期直接采用离线embedding，降低了模型压力。融合方式也不一样，简单的融合基本上没有参数引入**

## 是为啥会提出这么个东西，为了解决什么问题
- 问题：
  - 纯实时模型：如果只用近期行为（如RNN/Transformer），虽然响应快，但缺乏用户长期的、丰富的兴趣画像，且处理长序列计算成本高。
  - 纯批量模型：如果只用离线计算的用户嵌入（如PinnerFormer），虽然表征丰富、服务成本低，但无法捕捉用户最新的兴趣变化，推荐结果不够“鲜活”。
- 方法：
  - 融合短期和长期模型
  - 短期为了加速保证实时性，保证延迟降低，提出了一系列的工程上的优化加速
  - 长期为了保证兴趣丰富，同时利用更多的行为，直接采用离线训练好的embedding

## 为啥这个新东西会有效，有什么优势
- 对比大部分短期序列模型
  - TransAct本身的优势：
    - 强大的序列建模能力：Transformer能有效捕捉用户近期行为序列中的复杂模式和依赖关系。
      - 针对性的设计：实验发现位置编码在本任务中无效，因此被移除，体现了面向具体问题的优化。引入随机时间窗口掩码作为训练技巧，有效缓解了模型过度依赖近期行为导致的“信息茧房”（兔子洞效应），提升了推荐多样性。
    - 高效的特征融合：采用“连接法”进行早期融合，将候选物品信息与序列中每一个历史行为配对，实现了候选物品与用户历史兴趣的细粒度交互。
    - 融合压缩输出的方式：简单的直接提取前 k 个行为序列，保留了最近的兴趣同时降低了其它行为带来的噪音
- 对比其它长短期模型：
  - 混合架构的全局优势：
    - 实时流（TransAct）：提供敏捷性，捕捉即时兴趣，保证推荐的新鲜度。
    - 批量流（离线用户嵌入）：提供丰富性和经济性，承载长期兴趣，且计算成本低。
- 工程对比：
  - 提出了一系列基于 GPU 的 inference 加速方式。

## 与这个新东西类似的东西还有啥，相关的思路和模型
- 基于用户行为序列的模型：
  - BST，同样采用了 Transformer，但是BST强调行为顺序（使用位置编码），而TransAct实验发现位置信息无效
  - 其它短期行为序列模型（DIEN 之类），没有采用混合行为序列，更多的是单一行为。

## 在工业上通常会怎么用，如何实际应用
- 数据：
  - 混合行为序列，非常值得试一试，特别是 search 行为存在时间线
  - negative 行为可以考虑，但是可能要考虑一下怎么混合
- 短期：
  - TransAct：
    - transformer 架构值得试一试，对比 BST，早期融合机制非常值得试一试，可以对比把 候选 item 作为 query 去计算短期兴趣向量的方式
    - search 可能会存在时间位置的关系对于序列，可能需要思考一个方式把 timestamps 加入进去
    - 输出的压缩方式值得试一试，本质上是序列特征的pooling，直接保留前K个并且flatten后直接 concat 送入 DCN V2，可以试一试。这里的 K 需要根据业务进行调整。同时不一定要和 Transformer 合起来用，任何序列输出都可以使用。
- 长期：
  - 可以考虑直接引入 pretrained embedding
- loss 设置：
  - 可以参考对不同用户，不同行为在不同任务上的权重设置
- 工程：
  - 所有优化均可以尝试

## 参考
- https://zhuanlan.zhihu.com/p/652207249
- https://zhuanlan.zhihu.com/p/638012217




