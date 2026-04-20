# DCN^2 Interplay of Implicit Collision Weights and Explicit Cross Layers for Large-Scale Recommendation

# 标题
- 参考论文：DCN^2 Interplay of Implicit Collision Weights and Explicit Cross Layers for Large-Scale Recommendation
- 公司：Teads 
- 链接：https://arxiv.org/html/2506.21624v1
- Code：无，`待补充`
- `泛读`

# 内容

## 摘要
- DCN_V2是很强的baseline模型，能高效的学习简单的学习特征交叉，并且比很多模型（DeepFFM）需要的计算量小。
- 本文主要处理以下几个问题：
  - 表达能力受限，cross layer中间信息丢失，显式特征交互建模不够
  - embedding table 里面的特征碰撞

## 1 简介
- DCN_V2的局限：
  - 信息丢失：每一层的projection压缩可能会导致信息丢失
  - embedding table里面的碰撞管理
  - 显式特征交互建模信息不够
- 本文贡献：
  - 解决cross layer交叉中的信息丢失
  - 解决embedding table里面的碰撞问题
  - 解决算法偏见在显示的 pairwise interactions 建模

## 2 相关工作
- 模型方向
  - FM base 模型：
    - FM，DeepFM，DeepFFM
  - Attention base 模型：
    - AutoInt
  - GNN base 模型：
    - GraphFM
- 碰撞问题：same-representation multi-item occurrences
  - 加入 semantic IDs，high-level identifiers that preserve semantic meaning of items 
  - 参数化 hash space to mitigate collision impact

## 3 DCN^2 模型架构
<p style="text-align: center">
    <img src="./pics/DCN^2/DCN^2_3_模型结构.png">
      <figcaption style="text-align: center">
        DCN^2_模型结构
      </figcaption>
    </img>
    </p>

## 3.1.Collision-weighted lookups
引入碰撞权重感知碰撞，碰撞权重间接反映碰撞的影响：预测误差驱动：当多个特征值映射到同一桶（碰撞），该桶的嵌入向量需要同时表示多个特征的语义，可能导致预测误差增大（因为单一嵌入无法充分区分不同特征）。在训练过程中，梯度下降会调整该桶的碰撞权重（通常下调），以减少其对预测的过度影响。

## 3.2.onlydense layer as an alternative to Cross
- 原始的DCN_V2 Cross层采用通过低秩投影将输入映射到较低维度进行交互运算，随后再映射回原始维度。这种投影过程可能导致信息丢失，尤其是连续应用多个Cross层时（由于每次投影的压缩性质）。
- 作者发现我们可以通过直接在输入的原始维度空间（dense space）中进行特征交互，避免投影到低维空间，使用比较少的layer层数，从而减少信息丢失。同样可以达到相似甚至更好的模型效果。
- 具体公式如下：
  - x_t = α(W ⋅ x + b_0)
  - x_r = x_t ⊙ x ⋅ ϕ
  - α denotes the activation function (ReLU in practice)
  - W ∈ ℝ_d × d, d 是 embedding 维度
  - ⊙ 和 ϕ 是 scaled Hadamard product，实际中 ϕ 一般取值在 1 - 3 中间。
  - 注意这里并没有和x_0 进行点成，而是和上一层的输出和这一层投射之后的结果进行点成。
  - 这一点和DCN_V3里面的思路很像，本质上是抛弃了显性增长的特征交叉阶数，采用了指数增长。

## 3.3.Making pairwise interactions explicit
- 作者认为针对V2的Cross 层通过迭代投影或变换捕获交互，偏向于隐式（implicit）建模，可能无法充分模拟 FFMs 那样的显式成对交互。
- 提出SimLayer 针对输入嵌入（embeddings）计算每对特征的点积（dot product），捕获特征间的相似性（similarity），并通过激活函数和投影生成交互表示。
- 同时 SimLayer 共享权重矩阵，将参数量和计算复杂度降至可接受水平。
- 最后把 SimLayer 的输出作为一个额外的 logit，直接与 DCN 部分的输出融合，增强模型的预测能力，弥补 Cross/OnlyDense 层在显式交互上的不足。
- 这里借鉴了FFM的思路，对于每个embedding都会和其它embedding进行特征交叉点成，然后apply一个W的投影矩阵，作者没有放出来代码，`具体操作后续看看代码`
- 公式如下：
  - <p style="text-align: center">
      <img src="./pics/DCN^2/DCN^2_3.3_simlayer公式.png">
        <figcaption style="text-align: center">
          DCN^2_simlayer公式
        </figcaption>
        </img>
        </p>

## 4.Benchmarks on open data sets
- preprocess:
  - Log transform on continuous features
- 参数：
  - Batch size: 2500
  - learning rate: 0.0001 to 0.01
  - Adam’s beta one parameter：0.0 to 0.9
  - embedding dimension: 8 to 16

## 5.Hash space scaling laws
- With smaller hash space, performance drops for all approaches, even though less for collision-weighted lookups.
- With smaller hash space, more smaller weights are observed.

## 6.DCN2 online: 0.5 billion+ predictions per second and A/B results
- scaled后能达到多出 0.5 billion predictions per second
- CTR提升3.2% RPM，CVR提升4.2% swCR和0.37% GR，CPA降低2.8%。

## 7.Migrating from DCNv2 - an AutoML case study for win probability models
- 采用AutoML来进行整个开发流程
- embedding dimension increased from 6 in DCNv2 to 16 in DCN^2
- 没有做任何的feature combination，主要依赖DCN^2 的全新的 only-dense layer

## 8.DCN^2 Inference - lessons and battle scars
- [ ] 待细读

## 9 总结
- 总结
  - 在embedding碰撞问题上，对于高纬系数数据，达到了 richer embeddings and better predictions
  - 用全新的only-dense layer替换了原来的显性交叉层，以很少的参数达到了原先的效果，这里其实是assume原先的cross需要更多层
- 未来研究方向
  - 用更 aggressive collision weight schemes
  - 借鉴 LoRA based fine-tuning of existing models (with implications for transfer learning)
  - 推断可能训练一个更大的模型，然后采用LoRA的方式微调一个小模型，借鉴LLM的方式。


# 思考

## 本篇论文核心是讲了个啥东西
- 提出了一个处理embedding里面冲突的问题方案
- 提出了一个only-dense cross，可以不需要那么多layer但是可以实现4、5层DCN_V2 cross layer的效果甚至更好
- 提出了一个效果更好的显示交叉框架，独立在cross layer外，单独完成特征两两交叉的任务。

## 是为啥会提出这么个东西，为了解决什么问题
- 对比之前的DCN_V2
  - 提出only-dense layer，解决了信息丢失的问题
  - 提出only-dense layer，解决了用全维度的矩阵的话，参数太多模型太复杂的问题
  - 提出sim-layer解决了DCN_V2的cross层不够显性特征交叉的问题
- 对比大部分FM为base的模型和DCN的模型：
  - 之前大部分模型都会认为DCN的cross是显性交叉，但是最新的研究显示DCN的cross并不是显性交叉，指数增加更可能不是显性交叉，或者说不够显性。提出了用FFM的思路，提出了sim-layer，实现embedding-level的更明显的两两特征显性交叉。

## 为啥这个新东西会有效，有什么优势
- only-dense layer
  - 不使用projection dim，保留了全部信息，保证了不会信息丢失
  - 降低了layer数，减少了noise的引入，也减少了参数数量
  - 采用了和DCN_V3这篇paper一样的概念，不再线性增加交叉阶数，而是指数增加，这样同时兼具了高阶特征交叉，同时保证了参数不会加太多，同时不会引入过度的noise。
- sim-layer：
  - 之前大部分模型都会认为DCN的cross是显性交叉，但是很多paper的研究显示DCN的cross并不是显性交叉，指数增加更可能不是显性交叉，或者说不够显性。提出了用FFM的思路，实现embedding-level的更明显的两两特征显性交叉。
- dnn:
  - 同时保留了DNN层的隐性特征交叉，采用了stacked结构，把cross 和 DNN 进行了融合
- 总结，如果从Wide & Deep 的角度去解读：
  - Wide 部分相当于是sim-layer，保证了embedding-level的低阶显示特征交叉
  - Deep 部分相当于是指数高阶cross layer stacked 一个 DNN，保证了部分显示和隐式的高阶特征交叉，同时保留了纯隐式的特征交叉。并且采用了f_i * f_e 的结构。

## 与这个新东西类似的东西还有啥，相关的思路和模型
- embedding 碰撞的问题
  - 参考2 section
- cross layer特征交叉问题：
  - 参考DCN_V3里面的思路，提出了指数增加特征交叉
- embedding-level 显性特征交叉：
  - 所有FM base 的模型，这里采用了FFM

## 在工业上通常会怎么用，如何实际应用
- 目前为止还没看到论文放出的code。
- 本文实现的layer都可以考虑试一试，因为都不是特别复杂并且参数引入也不是很多。
- only-dense layer 同 DCN_V3一样可以尝试和线性cross一起或者直接替换掉线性cross。
- sim-layer 可以尝试加入，看看更显示的特征交叉会不会更好。
  









    