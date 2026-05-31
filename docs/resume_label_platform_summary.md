# Resume Packaging: Label Platform

## One-line Project Title

Model-in-the-loop AI Annotation Platform with Label Studio and MobileSAM

## Short Project Description

Built a local-first human-in-the-loop annotation platform integrating Label Studio with Dockerized ML services for model-assisted labeling. Extended the image segmentation workflow with MobileSAM pre-labeling, BrushLabels/RLE conversion, mask quality metadata, prompt-stability uncertainty, human correction delta tracking, and a lightweight scikit-learn correction-risk predictor.

## Strong Resume Bullets in English

- Built a local-first model-in-the-loop annotation platform integrating Label Studio with Dockerized ML backends for text classification, image classification, and MobileSAM-assisted image segmentation.
- Implemented a MobileSAM-compatible segmentation backend that outputs Label Studio BrushLabels/RLE masks with prompt metadata, mask quality instrumentation, and reproducible Docker smoke validation.
- Added prompt-stability uncertainty estimation by perturbing bbox prompts, generating multiple segmentation candidates, and measuring pairwise mask IoU and disagreement area.
- Developed a human correction delta dataset comparing model-generated masks with human-edited annotations using IoU, Dice, added/removed area, bbox alignment, centroid shift, and correction severity.
- Trained a lightweight scikit-learn LogisticRegression correction-risk predictor from human correction records to estimate whether model-generated segmentation masks require major human revision, using only pre-correction metadata to avoid target leakage.
- Built an active-review queue that combines correction-risk predictions, prompt-stability uncertainty, mask quality flags, geometry heuristics, and lightweight diversity signals to prioritize high-value segmentation samples for human review.
- Designed the system as a data-centric feedback loop for future active review prioritization while keeping MobileSAM as the validated lightweight runtime and SAM2 as future work.

## Strong Resume Bullets in Chinese

- 构建本地优先的 model-in-the-loop AI 标注平台，将 Label Studio 与 Docker 化 ML backend/trainer 服务集成，支持文本分类、图像分类和 MobileSAM 辅助图像分割标注。
- 实现 MobileSAM-compatible 图像分割预标注后端，输出 Label Studio BrushLabels/RLE mask，并附带 prompt 元数据、mask 质量检测和可复现的 Docker smoke validation。
- 增加 prompt-stability uncertainty 机制，通过扰动 bbox prompt、生成多组候选 mask、计算 pairwise IoU 和 disagreement area 来评估分割稳定性。
- 设计 human correction delta dataset，对比模型预标注 mask 与人工修正 mask，记录 IoU、Dice、增删面积、bbox 对齐、中心点偏移和修正严重程度。
- 基于人工修正记录训练轻量 scikit-learn LogisticRegression correction-risk predictor，用修正前 metadata 预测 mask 是否可能需要重大人工修改，并避免使用 delta 标签泄漏特征。
- 构建 active-review queue，融合 correction-risk 预测、prompt-stability 不确定性、mask quality flags、几何启发式特征与轻量多样性信号，对高价值图像分割样本进行人工复核优先级排序。
- 将系统设计为面向未来主动复核排序的数据闭环基础，同时保持 MobileSAM 为已验证运行时，SAM2 仅作为未来工作。

## Technical Keywords

- Label Studio OSS
- Docker Compose
- ML backend / trainer service
- Human-in-the-loop annotation
- Model-in-the-loop annotation
- MobileSAM
- BrushLabels / RLE mask
- Image segmentation
- Segmentation quality metadata
- Prompt-stability uncertainty
- Human correction delta dataset
- Correction-risk prediction
- Active-review queue
- scikit-learn LogisticRegression
- Data-centric AI
- Smoke tests and validation gates

## Honest Boundaries / What Not to Overclaim

- Do not claim this is a production-grade active learning platform.
- Do not claim full active learning acquisition is implemented.
- Say active-learning-style review prioritization, not a complete active learning system.
- Do not claim the queue automatically modifies or imports Label Studio tasks.
- Do not claim the system automatically determines label correctness.
- Do not claim MobileSAM is fine-tuned.
- Do not claim SAM2 is implemented or deployed.
- Do not present the correction-risk predictor as a production-quality model; it trains only when enough human correction delta records exist.
- Do not present deterministic image detection or text NER demo paths as fully trained production models.

## Interview Talking Points

- Why local-first: reproducible development, controlled data paths, simple Label Studio integration, and inspectable artifacts.
- How Label Studio integration works: task JSON, local file storage, ML backend prediction endpoint, BrushLabels/RLE masks, webhook-driven trainer flow.
- Why MobileSAM: lightweight validated segmentation runtime that can produce useful pre-labels without building a heavy segmentation training stack.
- Layer 1 design: mask quality metadata makes every prediction inspectable before adding more automation.
- Layer 2 design: prompt-stability uncertainty measures whether small prompt changes produce stable masks.
- Layer 3 design: human correction deltas convert annotation edits into supervised feedback data.
- Layer 4 design: correction-risk learning predicts major human revision risk from pre-correction metadata while avoiding target leakage.
- Layer 5 design: active-review queue ranks review candidates using correction risk, uncertainty, quality flags, geometry, and lightweight diversity without changing Label Studio tasks.
- Why this is not yet full active learning: there is no automatic acquisition loop, automatic task import, or model retraining from selected queue items.
- Engineering tradeoff: use lightweight, deterministic, testable components before introducing heavier model training or queue automation.
