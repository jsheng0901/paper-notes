# Efficient Long Sequential User Data Modeling for Click-Through Rate Prediction

# 标题
- 参考论文：Efficient Long Sequential User Data Modeling for Click-Through Rate Prediction
- 公司：Alibaba
- 链接：https://arxiv.org/pdf/2108.04468
- Code：https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/LongCTR/ETA/ETA.py
- 时间：2022
- `泛读`

# 内容

## 摘要
- 问题：
  - 同 ETA paper
- 方法：
  - 提出 ETA-Net，基于哈希的高效目标注意力（Target Attention）网络，通过低成本的位运算（Bit-wise operations）将标准注意力的复杂度降低了几个数量级。
  - 这里应该和 ETA 一样。这篇论文应该是 ETA 论文的扩展。

## 1 INTRODUCTION
- 同 ETA paper

## 2 RELATED WORK
- 同 ETA paper

## 3 PROBLEM DEFINITION
- 问题数学定义：
  - CTR 预估被建模为一个二分类问题，目标是预测用户 u 在特定特征集合 Xm 下点击目标物品 i 的概率 pm。
- 输入特征分类：
  - 典型的输入特征包含五类：用户画像、上下文特征、目标物品属性、短期行为序列和长期行为序列 (X_mlt)。
- 计算复杂度痛点：
  - 线性增长：标准目标注意力（Target Attention, TA）的复杂度为 O(L⋅d)，其中 L 是序列长度，d 是嵌入维度。
  - 在线压力：在真实推荐系统中，需要对 Nc 个候选物品进行打分，总复杂度飙升至 O(Nc⋅L⋅d)。
  - 性能牺牲：由于 Nc 和 d 通常很大，为了满足延迟要求，工业界不得不将序列长度 L 限制在 100 以内，这不可避免地降低了预估精度。
- 本论文目标：
  - 开发低于 O(L⋅d) 复杂度的算法
  - 提出可落地的系统架构
  - 并验证其有效性

## 4 METHODOLOGY
<p style="text-align: center">
    <img src="../../../pics/ETA-Net/ETA-Net_4_模型结构.png">
      <figcaption style="text-align: center">
        ETA-Net 模型结构
      </figcaption>
    </img>
  </p>

### 4.1 Model Overview
- 同 ETA paper

### 4.2 Efficient Target Attention Network
- 同 ETA paper

## 5 EXPERIMENTS
- 同 ETA paper

## 6 Conclusion
- 此篇paper完全就是ETA paper，只是换了一种写法更加formal


