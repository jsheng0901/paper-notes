# Deep Learning Recommendation Model for Personalization and Recommendation Systems

# 标题
- 参考论文：Deep Learning Recommendation Model for Personalization and Recommendation Systems
- 公司：Facebook
- 链接：https://arxiv.org/pdf/1906.00091
- Code：https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DLRM/src/DLRM.py
- `泛读`

# 内容

## 1 简介
有两个基本观点（primary perspectives）有助于个性化和推荐的深度学习模型的架构设计（architectural design）：
- 第一种观点来自推荐系统。
  - 最初采用内容过滤（content filtering），其中一群专家将产品划分为不同的类目（categories），用户选择他们喜欢的类目，而系统根据用户的偏好进行匹配。该领域随后发展为使用协同过滤（collaborative filtering），其中推荐是基于用户历史的行为，例如对item 的历史评分。
  - 邻域（ neighborhood）方法通过将用户和item 一起进行分组（grouping）从而提供推荐。 
  - 潜在因子（latent factor）方法通过矩阵分解技术从而使用某些潜在因子来刻画用户和item。
- 第二种观点来自预测分析（predictive analytics）。
  - 依靠统计模型根据给定的数据对事件的概率进行分类（classify）或预测（predict）。
  - 从简单模型（例如线性模型和逻辑回归）转变为包含深度网络的模型。
  - 为了处理离散数据（categorical data），这些模型采用了（embedding）技术从而将 one-hot 向量和 multi-hot 向量转换为抽象空间（abstract space）中的稠密表示（dense representation）。这个抽象空间可以解释为推荐系统发现的潜在因子空间（space of the latent factors）。
- 本文提出了一种全新的基于以上两种思路结合的模型：
  - 使用 embedding 来处理代表离散数据的稀疏特征（sparse features）
  - 使用多层感知机（multilayer perceptron: MLP）来处理稠密特征（dense features）
  - 使用 Factorization machines 中提出的统计技术显式的交互（interacts）这些特征
  - 最后，模型通过另一个 MLP 对交互作用进行后处理（post-processing）从而找到事件的概率。
- 此外本文还设计了一种特殊的并行化方案：
  - 该方案利用 embedding tables 上的模型并行性（model parallelism）来缓解内存限制（memory constraints）
  - 同时利用数据并行性（data parallelism）从全连接层扩展（scale-out）计算。

## 2 模型架构及设计
<p style="text-align: center">
    <img src="./pics/FiBiNET/DLRM_2_模型架构.png">
      <figcaption style="text-align: center">
        DLRM 模型架构
      </figcaption>
    </img>
    </p>

## 2.1 Components of DLRM
通过回顾早期模型，我们可以更容易理解 DLRM 的高级组件

### 2.1.1 Embeddings
利用 embedding tables 将离散特征映射到稠密表示

### 2.1.2 Matrix Factorization
从矩阵分解中可以回忆出，W和V矩阵可以解释为两个 embedding tables，其中每一行代表潜在因子空间（latent factor space ）中的 user/item。这些 embedding 向量的内积可以得出对评级有意义的后续预测，这是因子分解机（factorization machines）和 DLRM 设计的关键观察。

### 2.1.3 Factorization Machine
捕获不同 embedding 向量pair 对之间的交互，FM 显著降低了二阶交互的复杂度，从而产生线性计算复杂度。

### 2.1.4 Multilayer Perceptron
捕获更复杂的交互（interactions）。例如，已经表明：给定足够的参数、具有足够深度（depth ）和宽度（ width ）的 MLP 能够以任何精度（precision）拟合数据。这里MLP开始代替FM里面的内积方式计算特征embedding交叉。

## 2.2 DLRM Architecture
- 令用户和 item 使用很多连续（continuous）的、离散（categorical）的特征来描述。
  - 处理离散特征，每个离散特征将由相同维度的 embedding 向量来表示，从而推广了矩阵分解中潜在因子的概念。 
  - 处理连续特征，所有的连续特征整体被concat成一个特征向量后被一个 MLP 转换成同样维度为n的向量（这个 MLP 我们称之为 bottom MLP 或者 dense MLP），这将产生和 embedding 向量相同维度的 dense representation。注意这里所有连续特征整体才能得到一个 embedding。
- 根据 FM 处理稀疏数据的直觉，通过 MLP 来显式计算不同特征的二阶交互，这里通过计算所有 representation 向量之间的 pair 对的内积来实现的。
- 这些内积和连续特征 representation （即经过原始连续特征经过 bottom MLP 之后得到的）拼接起来，馈入另一个 MLP（我们称之为 top MLP 或者 output MLP）。
- 最终这个 MLP 的输出馈入到一个 sigmoid 函数从而得到概率
- 注意这里的top MLP包含：
  - 显式二阶交互特征
  - 连续representation 特征
  - 这里并没有馈入一阶 embedding 特征

## 2.3 Comparison with Prior Models
- 对比许多基于深度学习的推荐模型使用类似的基本思想来生成高阶项 term 从而处理稀疏特征。例如：Wide and Deep、Deep and Cross、DeepFM、xDeepFM 网络等等，它们设计了专用网络（specialized networks）来系统地构建高阶交互（higher-order interactions），DLRM 特别地模仿因子分解机，以结构化方式与 embedding 交互，从而在最终 MLP 中仅考虑 embedding pair 对之间的内积所产生的的交叉term，从而显著降低了模型的维度。本文认为，在其他网络中发现的、超过二阶的更高阶交互可能不一定值得额外的计算代价和内存代价。
- DLRM 和其它网络之间的主要区别在于：网络如何处理 embedding 特征向量及其交叉项 （cross-term）：
  - DLRM（以及 xDeepFM）将每个embedding 向量解释为表示单个类目（category）的单元（unit），交叉项仅在类目之间产生。也就是说这里的交叉是field-level的交叉，而不是bit-level。
  - 像 Deep and Cross 这样的网络将embedding 向量中的每个元素视为单元，交叉项在元素之间产生。因此，Deep and Cross 不仅会像 DLRM 那样通过内积在不同embedding 向量的元素之间产生交叉项，而且会在同一个embedding 向量内的元素之间产生交叉项，从而导致更高的维度。
  - 换句话来说，区别主要就是特征交叉的计算方式。


# 思考

## 本篇论文核心是讲了个啥东西
- 提出了一个全新的框架，从基础的矩阵分解到FM的二阶特征交叉，提出从不同特征角度，特征类似矩阵分解里面的两个向量，特征交叉可以适用于FM的二阶特征交叉。
- 提出一个数据并行和模型并行的思路，不过具体方法并没有放出来。
- 提出了一种对dense feature整体拼接后再转化成同样维度的vector的方法，实现了两种类型feature的后期特征交叉计算。

## 是为啥会提出这么个东西，为了解决什么问题
- 数据并行和模型并行解决了训练速度和memory的问题
- 本文认为高阶的特征交叉特意的设计不同的network是没有太多意义的浪费空间是时间，于是提出只用FM实现二阶的特征两两交叉，把高阶交给MLP来做。

## 为啥这个新东西会有效，有什么优势
- 本质上这个就是类似stacked结构的 DeepFM
- 文章结构很简洁，简单的二阶特征交叉，配上MLP来做隐性交叉，就可以满足当前的用户数据结构需求。

## 与这个新东西类似的东西还有啥，相关的思路和模型
- 对比FM类型的模型，基本上思路都一样。

## 在工业上通常会怎么用，如何实际应用
- 目前来看在Facebook有成功的落地，代码参考：https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/DLRM/src/DLRM.py
- 简单的FM加上MLP，外加单独对不同类型的feature做处理，可能是比较能借鉴的方向。

# 参考
- https://www.huaxiaozhuan.com/applications/recommendation/ctr_prediction/chapters/2019_DLRM.html