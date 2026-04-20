# Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction

# 标题
- 参考论文：Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/1905.09248
- Code：暂时没找到
- 时间：2019
- `泛读`

# 内容

## 摘要
- 问题：
  - 长序列的难点在于线上推理，随着序列变成会带来系统延迟和存储增加
- 方法：
  - 把用户兴趣计算 UIC (User Interest Center) 从模型中独立出来，不需要在推理时刻计算，没有 latency 线上推理的时候
  - 提出全新的 MIMN (Multi-channel user Interest Memory Network) 结构，来提取长序列的用户兴趣序列
- 本质上是把兴趣计算和 CTR 分离开，由于是独立模块，理论上可以不受到长度限制，scaling up 到 1000+ 长度可以，已经在阿里上线。并且是业界第一个把长序列上线的方法。

## 1 INTRODUCTION
- 问题：
  - 目前比较成功的方法主要是：
    - pooling-based 的架构：它将用户的历史行为视为独立的信号，并应用sum/max/attention 等池化操作来summarize 用户的兴趣representation。
    - sequential-modeling 架构：它将用户的历史行为视为序列信号，并应用 LSTM/GRU 操作来summarize 用户的兴趣representation。
  - 上面所有模型都需要存储在online serving 中，但是随着用户行为序列增加，系统延迟和存储成本随着用户行为序列的长度大致成线性关系。
  - DIEN 模型做了大量的工程优化，也只能处理 50 长度的用户行为序列，但是对于电商活跃用户两周内留下长度超过 1000 的行为（如点击、转化等）。
  - 并且随着用户行为的序列长度增加放进模型，offline 可以带来明显的 AUC 增长。
- 方法：
  - serving 系统的角度：
    - UIC: User Interest Center 用户兴趣中心的独立模块，将用户兴趣建模中最耗资源的部分和整个模型解耦。UIC 聚焦于在线serving 的用户行为建模问题，它维护每个用户的最新兴趣representation 。
    - UIC 的关键是用户的状态更新仅取决于实时的用户行为触发事件，而不依赖于流量请求。也就是说，UIC 对于实时 CTR 预估是无延迟的 latency free。
  - 机器学习算法的角度：
    - 鉴了 NTM 的记忆网络 memory network  的思想，并提出了一种叫做  MIMN: (Multi-channel user Interest Memory Network （多通道用户兴趣记忆网络）的新颖架构
      - 引入了记忆利用正则化（防止记忆单元浪费）和记忆诱导单元（更有效地读写和更新记忆），使其特别擅长从长序列中提炼和维持长期兴趣。
  - 理论上 UIC 和 MIMN 的共同设计方案使得模型能够处理无限长度的用户行为序列数据的用户兴趣建模。
- **主要共享**
  - 介绍了一个工程实践（hands-on practice）：结合学习算法和 serving 系统的协同设计来完成 CTR 预估任务。
  - 设计了一个新颖的 UIC 模块，它将沉重（ heavy ）的用户兴趣计算与整个 CTR 预估过程分离。UIC 对流量请求没有延迟，并允许任意复杂的模型计算，其中模型计算以离线模式工作。
  - 提出了一个新颖的 MIMN 模型，它改进了原有的 NTM 架构，具有记忆利用正则化memory utilization regularization 和记忆归纳单元 memory induction unit 两种设计，使其更适合用于兴趣的学习。MIMN 很容易用 UIC server 实现，其中 UIC server 增量地更新用户的兴趣representation 。

## 3 REALTIME CTR PREDICTION SYSTEM
<p style="text-align: center">
    <img src="../../pics/MIMN/MIMN_3_RTP_system.png">
      <figcaption style="text-align: center">
        RTP系统设计
      </figcaption>
    </img>
  </p>

### 3.1 Challenges of Serving with Long Sequential User Behavior Data
- 核心：
  - 90%的特征是用户行为特征
  - 用户行为特征随着长度的递增，offline AUC 明显提升
- 问题：
  - DIEN 的尝试已经优化了很多工程内容，但是最多接受用户行为序列长度为150，更不用说1000的长度
  - 存储约束：
    - 与用户数和序列长度线性相关。为6亿用户存储1000长度的序列需要6TB高性能内存，成本高昂。
  - 延迟约束：
    - 与序列长度线性相关。DIEN模型处理1000长度的序列时，500 QPS 延迟高达200ms，远超线上服务的可接受范围（通常要求 500 QPS < 30ms）。

### 3.2 User Interest Center
- 方法：
  - UIC 如上图 B：UIC server 为每个用户维护最新的兴趣representation
  - 用户状态的更新仅取决于实时用户行为触发事件，而不是取决于请求
  - UIC 处理1000长度的序列时，500 QPS 延迟降低到 19ms
- **本质上是离线处理用户行为和兴趣表达，也就是在特征交叉前把用户行为的表达通过另一个serve计算出来并且直接送进特征交叉。也就是TransAct 里面提到的另一种思路，不是end-to-end的模型设计思路。**

## 4 MULTI-CHANNEL USER INTEREST MEMORY NETWORK
- 用户行为长序列模型的具体设计思路

### 4.1 Challenges of Learning From Long Sequential User Behavior Data
- 问题：
  - RNN（如RNN、GRU、LSTM）处理长序列效果并不好，遗忘性：其隐藏状态是为预测下一个目标而设计的，并非为长期记忆而优化，会导致长期信息丢失。话句话说就是被近期行为主导，导致长期信息丢失。
  - 注意力机制的效率瓶颈：虽然能捕捉相关性，但必须在线存储所有原始序列，且计算开销随序列长度线性增长，在工业场景中（序列长达数千）无法承受。
- 方法：
  - 左子网络聚焦于用户行为序列的用户兴趣建模
  - 右子网络遵循传统的 Embedding &MLP 范式，左子网络的输出以及其它特征作为输入。
  - 提出MIMN：
    - UIC 存储MIMN的外部记忆张量，并根据用户的新行为进行更新。这样，UIC可以增量式地从用户行为序列中捕获其兴趣。
    - 记忆正则化：
      - 通过提高记忆利用率来增强UIC中记忆张量的表达能力
    - 记忆诱导单元：
      - 捕捉用户兴趣中更复杂的高阶模式和非线性演化

### 4.2 Neural Turing Machine
<p style="text-align: center">
    <img src="../../pics/MIMN/MIMN_4.2_模型结构.png">
      <figcaption style="text-align: center">
        模型结构
      </figcaption>
    </img>
  </p>

- 标准的 NTM 通过记忆网络从序列数据中捕获并存储信息
- 在 time step t，记忆（memory ）的参数记作 M_t，M 个 记忆槽 memory slot
- 记忆读取 memory read 和 记忆写入 memory write ，通过一个控制器controller 来和记忆交互
- memory read：
  - 给定第 t 个行为的 embedding 向量，控制器根据当前行为生成一个“查询” k_t，
  - 通过软注意力机制（基于余弦相似度）计算 k_t 与每个记忆槽 M_t(i) 的相关性权重 w^r_t(i)。模拟了“回想”相关记忆的过程。
  - 对记忆槽进行加权求和，得到当前时刻的记忆摘要向量 r_t，浓缩了与当前行为相关的长期兴趣信息。
  - **本质上就是计算当前时刻对于之前所有记忆的加权，也就是长期兴趣的加权压缩**
- memory write：
  - 控制器生成写入权重 w^w_t、添加向量 a_t 和擦除向量 e_t。
  - 更新机制：
    - 擦除：E_t = w^w_t ⊗ e_t 决定从哪些记忆槽中抹去多少旧信息。
    - 添加：A_t = w^w_t ⊗ a_t 决定向哪些记忆槽添加多少新信息（来自当前行为）。
    - 合并：M_t = (1 - E_t) ⊙ M_{t-1} + A_t，通过先擦后加的方式，增量式、选择性地更新记忆体，实现对新知识的吸收和对旧知识的遗忘。
    - **这个本质上和 LSTM 的门思路很像，更新遗忘添加然后合并。区别是 LSTM 对于每个时刻都会做，也就是会出现遗忘很久之前的，这里有多个记忆槽来存储记忆，更不容易出现遗忘问题**
- 核心是先 write 来更新用户的新行为带来的兴趣变化，然后 read 在预测时来提取最新地相关信息，达到固定大小的存储空间（记忆槽个数是控制固定的 M 个），持续学习和维护用户的长期兴趣。

### 4.3 Memory Utilization Regularization
- 问题：
  - basic NTM 会遭受记忆使用不平衡的问题，尤其是在用户兴趣建模的场景下。热门的 item 倾向于在用户行为序列中频繁出现，并且主导着 memory 的更新，从而使得 memory 的使用变得低效。
- 方法：
  - 动态重平衡：
    - 引入一个可学习的权重转移矩阵 P_t。根据每个记忆槽的历史累积使用情况（g_t），动态调整当前的原始写入权重 w^w_t。
    - g_t = Σ_{c=1}^t w^{w̃}_c，为截至第t个时间步的累计更新权重，其中 w^{w̃}_c 表示第c个时间步中重平衡后的写入权重
    - P_t = softmax(W_g * g_t)            (5)
    - w^{w̃}_t = w^w_t * P_t               (6)
  - 正则化损失：
    - 设计损失函数 L_reg，其目标是使所有记忆槽的长期累积更新权重 w^{w̃} 的方差最小化。这相当于施加了一个软约束。
    - w^{w̃} = Σ_{t=1}^T w^{w̃}_t              (7)
    - L_reg = λ * Σ_{i=1}^m (w^{w̃}(i) - (1/m)Σ_{i=1}^m w^{w̃}(i))^2   (8)
  - **本质上就是让每个记忆槽都均匀的使用起来**

### 4.4 Memory Induction Unit
- 问题：
  - NTM的记忆体擅长存储“事实”（原始行为信息），但缺乏对信息间动态关联和趋势的提炼，例如各部分兴趣的动态演变过程。
- 方法：
  - 设计了一个MIU
    - k个通道选择：根据NTM的读取注意力权重 w^r_t，智能聚焦于当前最相关的k个兴趣通道。
    - 归纳更新：GRU单元。它的输入融合了三方面信息：
      - S_t(i) = GRU( S_{t-1}(i), M_t(i), e_t )
      - 自身上一状态 S_{t-1}(i)：表示该兴趣通道过去的演化历史。
      - NTM的当前记忆 M_t(i)：提供了该兴趣最新的“事实”。
      - 当前行为 e_t：提供了最新的外部刺激。
  - 通过GRU的时序建模能力，MIU能够持续地、增量地更新每个兴趣通道的演化状态，从而刻画兴趣的兴衰、深化或转移过程。
  - NTM负责记住用户“喜欢过什么”，而MIU则试图理解用户对这些事物的兴趣“是如何变化的”
  - 工程优势：
    - 所有兴趣通道共用同一个GRU的参数，没有参数增加
  - **本质上是结合了GRU和记忆槽，同步更新兴趣演化过程并且学习记录在多个记忆槽中**

### 4.5 Implementation for Online Serving
<p style="text-align: center">
    <img src="../../pics/MIMN/MIMN_4.5_线上服务器设计.pngg">
      <figcaption style="text-align: center">
        线上服务器设计
      </figcaption>
    </img>
  </p>

- 建议将UIC server 和 MIMN 的 co-design 的解决方案应用于具有以下条件的应用程序：
  - 丰富的用户行为数据
  - 以及实时用户行为事件的流量规模不能明显超过实时 CTR 预测请求的流量规模

## 5 EXPERIMENTS

### 5.1 Datasets and Experimental Setup
- Amazon Dataset： 长度截断到 100
- Taobao Dataset： 长度截断到 200
- Industrial Dataset： 
  - 前49天 train
  - 第 50 天 test
  - 包含 user 60 天的所有行为
  - 长度截断到 1000

### 5.2 Competitors & 5.3 Results on Public Datasets
- MIMN的效果最好，并且明显提升 AUC
- 证实了长用户行为包含了用户兴趣是多样的，且随着时间而动态演化

### 5.4 Ablation Study
- memory 的 slot 数量：
  - 用户行为序列的长度越长，需要的 slot 数量越多，如果太多了会出现 slot 的利用不平衡，学习不足，也就是过拟合
- Memory Utilization Regularization：
  - 明显提高每个 slot 的使用率，同时提升了 AUC，解决了上面问题

### 5.5 Results on Industrial Dataset
- MIMN 和 UIC server 的 co-design 对比 DIEN 能在不同行为序列长度保持恒定的延迟和吞吐量
- DIEN 只能到 50，然而 MIMN 可以到数千个的、长的用户行为序列数据
- A/B test CTR 和 RPM（Revenue Per Mille 每千次收入）均提高了 7.5%
- **本质上是把长用户行为做到了离线处理，保证了 latency free，但是同时保证了离线 serve 更新频繁，可以学习到当下的用户行为兴趣变化**

### 5.6 Practical Experience For Deployment
- UIC Server 和 RTP Server 的同步synchronization：
  - 模型部署被设计为每小时执行一次
  - MIMN学习的用户兴趣的稳定表示stable representation，从而使得 MIMN 具有良好的泛化性能
- 超大规模big-scale 数据的影响：
  - 例如双11的促销数据，样本的分布以及用户行为和日常情况大相径庭
  - 实验显示对于长期用户兴趣行为，最好移除 big-scale 数据
- Warm Up Strategy：
  - UIC 旨在进行增量更新，但是需要一开始有稳定的积累
  - 使用 120 天的历史行为（用户行为序列的平均长度为 1000）离线训练好的 MIMN 来推断，再将累计的 memory 进行增加更新
- Rollback Strategy：
  - 每天00:00 学到的用户兴趣representation 副本存储起来，并保存最近 7 天的副本。
  - 保证了出意外也不会出现模型训练作弊的情况

## 6 CONCLUSIONS
- 提出了一个工业级别的长序列解决落地的方案，核心还是如何解决存储和latency的问题
- 提出了UIC serve 和 增量模型 MIMN 算法
  - 系统架构创新：UIC服务器，把兴趣计算从线上 serving 中剥离，实现了对线上请求的零延迟影响。
  - 算法模型创新：基于记忆的MIMN模型，通过外部记忆网络和增量更新机制，以恒定存储成本处理理论上无限长的行为序列。
- **本质上如果把MIMN的长度换成DIN或者DIEN，可能模型并没有优势（记忆槽有限共享参数，并且不是end-to-end），但是工程上实现不了这个长度，本质是也就是工程上的成功来无限接近离线效果**


# 思考

## 本篇论文核心是讲了个啥东西
- UIC+MIMN 的、算法与系统协同设计的工业级解决方案，旨在攻克推荐/广告系统中对超长用户行为序列（长度可达1000甚至更长）进行建模的难题。
- 问题：
  - 传统序列模型（如RNN、DIEN）在处理长序列时，面临在线存储成本爆炸式增长和推理延迟线性增加的工程瓶颈。
- 方案： 
  - 系统架构革新：UIC模块。独立的用户兴趣中心服务。把用户兴趣计算从实时的CTR预测请求中分离出来，以异步、增量的方式更新用户兴趣状态，从而对高并发的线上广告请求实现 “零延迟” 影响。
  - 算法模型革新：MIMN模型。改进的记忆网络。将用户的长期兴趣压缩存储在一个固定大小的外部记忆矩阵中，以增量方式模拟对无限长行为序列的学习，从而以恒定存储成本维护用户兴趣。

## 是为啥会提出这么个东西，为了解决什么问题
- 问题：
  - 数据的价值：
    - 使用1000长度的行为序列，相比100长度，能为基础模型带来0.6%的AUC显著提升
  - 存储问题：
    - 为6亿用户存储长度为150的行为序列已需1TB高性能内存。若长度增至1000，则需6TB。
  - 推理延迟：
    - 即使经过深度优化，像DIEN这样的模型在处理1000长度序列时，延迟会从150长度的14ms飙升至200ms，完全无法满足在线服务通常要求的30ms以内延迟。
- 方法：
  - UIC 离线独立 serve，解决了延迟
  - MIMN 使用固定 memory slot，来存储用户长期兴趣，解决了存储问题
  - 合在一起保证了使用 1000+ 以上的用户行为序列

## 为啥这个新东西会有效，有什么优势
- 对比大部分序列模型：
  - 离线 UIC，解决延迟
  - MIMN 固定记忆槽，解决存储问题
  - 改进NTM，设计MINU为推荐系统场景使用：
    - 记忆利用正则化：
      - 解决基础记忆网络利用率不均的问题（热门行为霸占记忆槽），通过正则化引导模型在长期内更均衡地使用所有记忆单元，最大化固定容量下的信息承载量。
    - 记忆诱导单元：
      - 引入一个轻量的GRU网络作为“高阶处理器”，专门捕捉兴趣本身的演化过程，使模型不仅能记住“喜欢过什么”，还能理解兴趣“如何变化”。

## 与此论文类似的东西还有啥，相关的思路和模型
- 基于长序列的模型：
  - 对比SIM, UBR
- 基于用户兴趣分离模型：
  - 对比 TransAct，通过极致的工程优化（GPU、Triton）重新合并实现端到端的超长序列实时处理

## 论文有什么可以改进的地方，可以后续继续拓展研究
- 模型表达能力：
  - 记忆交互：
    - MIMN中不同记忆通道（兴趣）之间是相对独立的。未来可以探索记忆通道间的显式交互机制，以建模用户兴趣的交叉与融合（例如，“喜欢登山”和“喜欢摄影”可能共同影响对“户外相机”的兴趣）。
    - 这里需要思考，参数共享对于信息的丢失问题
  - 更强大的序列归纳器：
    - MIU中的GRU相对简单。可以探索用更强大的序列模型（如轻量级Transformer） 作为归纳单元，以捕捉更复杂的长期依赖和兴趣演化模式。
- 系统与效率：
  - 记忆压缩与量化：
    - 在固定大小的记忆矩阵基础上，能否对每个记忆槽的内容进行进一步压缩或量化，在几乎不损失效果的前提下，进一步降低存储和传输开销。
  - 更新频率与实时性权衡：
    - UIC的异步更新策略存在微小的“状态滞后”。可以研究更智能的更新触发策略，或多级兴趣状态缓存，在成本和实时性间取得更优平衡。
- 训练范式：
  - 端到端联合训练：
    - UIC/MIMN与下游CTR模型通常是分开训练或分阶段训练的。探索更紧密的端到端联合优化，可能使兴趣表征更适配最终任务。
    - 这里和TransAct 思路一致，需要解决工程问题
  - 无监督/自监督预训练：
    - 利用海量用户行为序列，对MIMN进行大规模无监督的预训练，学习通用的用户兴趣表示，再迁移到具体的推荐任务上。
    - 这里本质上就是离线学习用户兴趣表达的 embedding

## 在工业上通常会怎么用，如何实际应用
- UIC 的独立模块，可以尝试，但是需要权衡是走TransAct 路线，直接优化算法和服务器还是离线 UIC 的独立服务器设计。
- MIMN 里面的思路可以借鉴一下：
  - 记忆利用正则化，可以在MoE的多序列融合或者NLP里面transformer思路一样借鉴
  - GRU 的思路来学习兴趣的演化过程也可以借鉴，不过可以直接借鉴 DIEN 里面的 GRU 思路一样

## 参考链接
- https://www.huaxiaozhuan.com/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/chapters/9_ctr_prediction8.html


