# Resume Project: Label Platform

## 中文简历版本

**Label Studio + MobileSAM 人机协同图像分割标注平台**  
个人项目｜2026.05 - 2026.06  
技术栈：Python, FastAPI, Docker Compose, Label Studio OSS, MobileSAM, scikit-learn, NumPy, COCO/LVIS/DIS5K/CAMO/COD10K

- 构建本地化 human-in-the-loop AI 标注平台，集成 Label Studio OSS、Docker 化 ML backend/trainer 服务，支持 MobileSAM 辅助图像分割预标注与人工修正闭环。
- 实现 MobileSAM-compatible 图像分割预测后端，输出 Label Studio BrushLabels/RLE mask，并记录 prompt metadata、mask area、bbox alignment、RLE validity、border-touching 等质量特征。
- 设计 prompt-stability uncertainty 机制，通过扰动 bbox prompt 生成多组候选 mask，计算 pairwise IoU、minimum IoU 与 disagreement area，用于识别不稳定分割结果。
- 构建 human correction delta dataset，对比模型预标注 mask 与人工/GT 修正 mask，记录 IoU、Dice、增删面积、bbox 偏移、centroid shift 与 major correction severity。
- 基于修正前 metadata 训练轻量 scikit-learn LogisticRegression correction-risk predictor，预测 mask 是否可能需要重大人工修改，并避免使用 GT/delta 字段造成目标泄漏。
- 构建 active-review queue，融合 correction risk、uncertainty、quality flags、geometry heuristics 与 diversity signal，对高价值图像分割样本进行人工复核优先级排序。
- 在 COCO、LVIS、DIS5K、CAMO、COD10K 等公开分割数据集上构建跨数据集验证流程，评估 MobileSAM 预标注 mask 的边界/形状风险信号，并通过 AP、ROC-AUC、Lift@20%、Precision@20% 等指标验证其对重大人工修正样本的识别能力，为后续 shadow-only review prioritization 提供实验依据。
- 设计 boundary/shape learned signal 的 shadow-only rollout 方案，保证实验分数不影响默认 Layer 5 排序，并通过 default-field equality、ordering equality、leakage guard、drift monitoring 与 safe review packet 控制上线风险。
- 组织 COCO 与 LVIS balanced human review pilots，并用 GT-derived IoU/Dice 构建 size-stratified correction effort summary；结果显示整体 learned signal 跨数据集不稳定，但 COCO/LVIS 小目标是明确高风险 slice，Major/Redo 分别达到 57.7%/51.7%，因此保持 shadow-only 并转向 size-aware calibration。

## 英文简历版本

**Label Studio + MobileSAM Human-in-the-loop Image Segmentation Platform**  
Personal Project｜May 2026 - Jun 2026  
Tech Stack: Python, FastAPI, Docker Compose, Label Studio OSS, MobileSAM, scikit-learn, NumPy, COCO/LVIS/DIS5K/CAMO/COD10K

- Built a local-first human-in-the-loop annotation platform integrating Label Studio OSS with Dockerized ML backend/trainer services for MobileSAM-assisted image segmentation and human correction workflows.
- Implemented a MobileSAM-compatible segmentation backend that returns Label Studio BrushLabels/RLE masks with prompt metadata, mask area, bbox alignment, RLE validity, border-touching, and other quality signals.
- Designed prompt-stability uncertainty estimation by perturbing bbox prompts, generating multiple candidate masks, and measuring pairwise IoU, minimum IoU, and disagreement area.
- Built a human correction delta dataset comparing model pre-label masks with human or GT-corrected masks using IoU, Dice, added/removed area, bbox shift, centroid shift, and major-correction severity.
- Trained a lightweight scikit-learn LogisticRegression correction-risk predictor using pre-correction metadata to estimate whether a segmentation mask may require major human revision while avoiding GT/delta target leakage.
- Built an active-review queue combining correction risk, uncertainty, quality flags, geometry heuristics, and diversity signals to prioritize high-value segmentation samples for human review.
- Created a cross-dataset validation workflow across COCO, LVIS, DIS5K, CAMO, and COD10K to evaluate boundary/shape risk signals for MobileSAM pre-label masks, using AP, ROC-AUC, Lift@20%, and Precision@20% to measure major-correction discovery.
- Designed a shadow-only rollout path for learned boundary/shape scoring, ensuring experimental scores do not affect default Layer 5 ordering through default-field equality checks, ordering equality checks, leakage guards, drift monitoring, and safe review packet export.
- Ran COCO and LVIS balanced human review pilots and built a GT-derived IoU/Dice size-stratified correction-effort summary; the learned signal was not uniformly stable across datasets, but small objects emerged as a clear high-risk slice on COCO/LVIS with 57.7%/51.7% Major/Redo rates, so the signal remained shadow-only while moving toward size-aware calibration.

## 推荐压缩版

如果简历空间有限，可以保留下面 4 条：

- 构建 Label Studio + Docker + MobileSAM 的本地化 human-in-the-loop 图像分割标注平台，实现 BrushLabels/RLE 预标注、人工修正、质量记录与验证闭环。
- 设计 mask quality、prompt-stability uncertainty 与 human correction delta 数据结构，将模型预标注和人工修正转化为可学习、可审计的反馈数据。
- 基于修正前特征训练 correction-risk predictor，并构建 active-review queue，融合风险、不确定性、几何特征与多样性信号，对高价值样本进行复核排序。
- 在 COCO/LVIS/DIS5K/CAMO/COD10K 上验证 boundary/shape 风险信号，并通过 GT-derived correction effort 发现 COCO/LVIS 小目标 Major/Redo 明显高于整体；结论保持 shadow-only，并将后续方向收敛到 size-aware risk calibration。

## 不建议写法

- 不建议写“生产级主动学习平台”，当前更准确的是“active-learning-style review prioritization”。
- 不建议写“已上线 SAM2”，仓库中 SAM2 只是 future work，已验证运行时是 MobileSAM。
- 不建议写“人物多类别语义分割/人体解析”，当前 Label Studio 配置是通用 `Object` mask，可描述为人物/前景对象分割场景。
- 不建议写“自动判断标签正确性”，系统是复核排序与风险提示，不是自动判定人工标签真伪。
- 不建议写“已通过真实生产 A/B 验证”，目前有 benchmark、shadow rollout 和小样本校准，但默认排序未提升。
