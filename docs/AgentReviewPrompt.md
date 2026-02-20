你是一个资深多模态机器学习系统工程师 + 研究代码 reviewer。

我要你对当前代码库进行一次【工程级 + 科学级】的全面评估，并在必要时直接给出修改方案。

=== 背景与约束 ===
- 本项目用于多模态对齐（contrastive learning）
- 支持模态：text / image / audio / video
- 视频在进入模型前已被分解为：
  - IMAGE: 16 帧 (16, 3, 224, 224)
  - AUDIO: 音频特征
- encoder 已冻结，仅训练对齐头 / adapter
- 不存在独立 video encoder / video head
- 损失函数为 InfoNCE，对应模态对之间的对齐
- 缺失模态通过 modality_masks 控制
- 单帧 image 在 batch 中可能被复制为 16 帧以适配 encoder

=== 你的任务 ===

请从以下【四个维度】对代码进行评估，并在必要时直接给出修改建议或示例代码。

---

### 1️⃣ 完整性评估（Correctness & Coverage）

请检查并回答：

- 当前代码是否 **完整覆盖** 以下情况：
  - 单模态样本（仅 text / image / audio / video）
  - 任意两模态组合
  - video 有 / 无 audio
  - batch 中混合单帧 image 和 16 帧 video
  - batch 中所有样本缺失某一模态
- modality_masks 是否在：
  - dataset → collate → encoder → loss
  全流程保持一致语义
- 是否存在 mask=True 但 feature 实际为零向量的情况
- 是否存在 mask=False 但 feature 被用于 loss 的情况

👉 请明确指出：
- 已覆盖 / 未覆盖
- 潜在 silent bug（不会报错但语义错误）

---

### 2️⃣ 工程性评估（Engineering & Performance）

请重点检查：

- Dataset 阶段是否存在：
  - 过重 I/O（如每 epoch ffmpeg 解码）
  - 不必要的 numpy ↔ torch 往返
- collate_fn 是否存在：
  - 不必要的 tensor copy / expand
  - 隐式语义变换（如单帧 → 16 帧）
- 是否存在：
  - batch 内重复推断 modality mask
  - 重复计算 encoder embedding
- 内存峰值是否会随 batch size / video 比例非线性增长

👉 请明确指出：
- 哪些地方 **在真实训练中一定会出问题**
- 哪些是“现在能跑，但 scale 会炸”的设计

---

### 3️⃣ 科学性评估（Modeling & Learning Correctness）

请从研究角度评估：

- 当前对比学习目标是否满足 InfoNCE 假设
- 视频样本分解为 image + audio 后：
  - 是否存在信息重复或 bias
  - 是否引入隐式 sample reweighting
- 单帧 image 复制为 16 帧是否：
  - 数学等价
  - 是否在 encoder / attention 中产生非预期行为
- 是否存在：
  - 实际无贡献但会影响梯度的 loss 项
  - 看似“增强约束”但其实冗余的设计

👉 请指出：
- 可以删掉而不影响学习效果的逻辑
- 科学上站不住脚的实现

---

### 4️⃣ 复杂度与冗余评估（Maintainability）

请检查并指出：

- 是否存在历史遗留逻辑（如 video triplet loss、deprecated feature key）
- 是否有可以：
  - 合并的分支
  - 抽象成 adapter 的 hack
- 是否存在：
  - “为了兼容而兼容”的代码
  - 实际 never triggered 的分支

👉 请给出：
- 可直接删除的代码清单
- 推荐的最小化实现版本（不需要完整重写）

---

### 5️⃣ 输出要求（非常重要）

你的输出请严格包含以下部分：

1. **总体结论**（是否可以进入长期训练 / 扩展）
2. **必须修改的问题（Critical）**
3. **建议修改的问题（Recommended）**
4. **可以删除的冗余代码**
5. **示例修改 / patch（如伪代码或 diff）**

⚠️ 请避免空泛建议（如“可以考虑优化”）
⚠️ 请基于当前代码现实，而不是理想重构

你的目标是：让这套系统 **稳定、可解释、可长期维护**。
