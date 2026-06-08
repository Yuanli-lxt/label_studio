# Auxiliary Notes: Label Platform Resume Explanation

这份文档用于面试前准备，解释简历中每一条项目经历背后的具体工程内容、可讲证据、边界和推荐说法。它不是简历正文，可以比简历更详细。

## 项目一句话

这是一个基于 Label Studio OSS 和 MobileSAM 的本地化 AI 标注平台，重点解决图像分割预标注、人工修正、质量评估、风险排序和实验验证闭环。

更口语化的面试说法：

我不是只接了一个模型预测接口，而是围绕“模型预标注之后，哪些样本最值得人去修、怎么记录修正、怎么避免实验信号污染线上排序”做了一套数据闭环。

## 1. 本地化 HITL 标注平台

简历原句：

构建本地化 human-in-the-loop AI 标注平台，集成 Label Studio OSS、Docker 化 ML backend/trainer 服务，支持 MobileSAM 辅助图像分割预标注与人工修正闭环。

详细解释：

这个项目把 Label Studio 作为标注 UI 和任务管理入口，把模型服务和训练/反馈服务拆成 Docker 化组件。Label Studio 负责展示图像、接收 mask 编辑和 review outcome；ML backend 负责模型预标注；trainer 服务负责接收 webhook、抽取人工修正数据、生成后续训练或评估数据。

可以强调的点：

- 本地优先部署，方便在 WSL2/Docker Desktop 中复现。
- Label Studio 通过 local-files 访问本地图像，不依赖外部云存储。
- 分割之外也保留文本分类、图像分类路径，但简历重点应放在图像分割。

边界：

- 不是 SaaS 多租户平台。
- 不是大规模生产标注平台。
- 更准确地说是本地可复现的 HITL 原型和实验平台。

## 2. MobileSAM 分割预标注后端

简历原句：

实现 MobileSAM-compatible 图像分割预测后端，输出 Label Studio BrushLabels/RLE mask，并记录 prompt metadata、mask area、bbox alignment、RLE validity、border-touching 等质量特征。

详细解释：

Label Studio 的分割标注需要 BrushLabels/RLE 格式，而 MobileSAM 输出的是模型侧 mask。这个项目实现了模型输出到 Label Studio 可编辑 mask 的转换，同时把 prompt bbox、mask bbox、RLE 长度、mask 面积比例、是否贴边、prompt 与 mask bbox IoU 等字段写入 metadata。

可以强调的点：

- 解决了“模型能分割”和“Label Studio 能展示、能编辑、能回收数据”之间的格式桥接。
- 对 RLE 和 mask 质量做了 smoke validation，避免空 mask、坏 RLE 或坐标错位直接进入标注流。
- MobileSAM 是已验证运行时；SAM2 在仓库中属于 future work，不应混淆。

边界：

- 没有 fine-tune MobileSAM。
- 当前标签配置是通用 `Object`，可用于人物/前景对象场景，但不是人体解析专用标签体系。

## 3. Prompt-stability uncertainty

简历原句：

设计 prompt-stability uncertainty 机制，通过扰动 bbox prompt 生成多组候选 mask，计算 pairwise IoU、minimum IoU 与 disagreement area，用于识别不稳定分割结果。

详细解释：

SAM 类模型很依赖 prompt。对于同一目标，如果 bbox 稍微扩大、缩小或平移，输出 mask 变化很大，说明这个样本对 prompt 敏感，人工复核价值更高。项目中通过生成多个 bbox prompt variants，计算候选 mask 之间的 pairwise IoU、min IoU 和 disagreement area ratio，形成稳定性评分。

可以强调的点：

- 这是不依赖 GT 的不确定性估计方法。
- 它适合预标注场景，因为生产时通常没有真实 mask。
- 能把“模型可能不稳”的样本提前送入 review queue。

边界：

- 它不是贝叶斯不确定性。
- 它不能直接证明 mask 错，只能说明 prompt 扰动下输出不稳定。

## 4. Human correction delta dataset

简历原句：

构建 human correction delta dataset，对比模型预标注 mask 与人工/GT 修正 mask，记录 IoU、Dice、增删面积、bbox 偏移、centroid shift 与 major correction severity。

详细解释：

如果只保存最终人工 mask，就很难知道模型错在哪里。这个项目把模型预标注 mask 和人工修正后的 mask 做差，提取 IoU、Dice、added area、removed area、bbox shift、centroid shift 等指标，并把修正程度归类为 major/minor/no-fix 等信号。

可以强调的点：

- 把人工修正转化为监督学习数据。
- 让后续 risk predictor 可以学习“哪些预标注特征会导致重大修正”。
- 在公开 benchmark 中也可用 GT 模拟人工修正，便于快速评估。

边界：

- 公共数据集 GT 模拟不等同于真实标注员行为。
- 人工修正数据量不足时，模型训练应跳过或标记为实验。

## 5. Correction-risk predictor

简历原句：

基于修正前 metadata 训练轻量 scikit-learn LogisticRegression correction-risk predictor，预测 mask 是否可能需要重大人工修改，并避免使用 GT/delta 字段造成目标泄漏。

详细解释：

项目中使用修正前可获得的 metadata，例如 mask_quality、prompt stability、几何特征和 review flags，训练一个轻量 LogisticRegression 模型预测 major correction risk。关键设计是避免使用 IoU、Dice、GT mask、correction severity 等只有评估后才知道的字段作为输入，否则会造成 target leakage。

可以强调的点：

- 模型轻量、可解释，适合作为 review queue 的辅助排序信号。
- 训练前检查最小样本数、正负样本数，不满足就跳过，避免伪训练。
- 输出 feature names、metadata 和训练报告，方便审计。

边界：

- 不应说这是生产级风险模型。
- 它是轻量实验模型，目的是验证数据闭环可行性。

## 6. Active-review queue

简历原句：

构建 active-review queue，融合 correction risk、uncertainty、quality flags、geometry heuristics 与 diversity signal，对高价值图像分割样本进行人工复核优先级排序。

详细解释：

review queue 的目标不是把所有样本都交给人，而是把最可能需要重大修正、最值得复核的样本排到前面。默认 Layer 5 排序融合 correction risk、prompt-stability uncertainty、rule-based quality score、geometry complexity 和 diversity score。

可以强调的点：

- 默认权重中 correction risk、uncertainty、quality、geometry、diversity 分开管理，便于消融。
- diversity 让队列不要只集中在同一类样本。
- queue 输出 rank、priority score、priority bucket 和 review reasons，可解释性较强。

边界：

- 这是 active-learning-style prioritization，不是完整 active learning acquisition loop。
- queue 不会自动修改 Label Studio 任务，也不会自动触发模型重训。

## 7. 跨数据集 boundary/shape 风险验证

简历原句：

在 COCO、LVIS、DIS5K、CAMO、COD10K 等公开分割数据集上构建跨数据集验证流程，评估 MobileSAM 预标注 mask 的边界/形状风险信号，并通过 AP、ROC-AUC、Lift@20%、Precision@20% 等指标验证其对重大人工修正样本的识别能力，为后续 shadow-only review prioritization 提供实验依据。

详细解释：

项目不仅在单个 demo 数据集上跑通，而是用多个公开数据集覆盖不同难点：COCO/LVIS 偏实例和长尾类别，DIS5K 偏高精细边界，CAMO/COD10K 偏伪装和低对比目标。通过这些数据集观察 boundary density、boundary complexity、component count、hole count、area ratio、extent、touches border 等预测时可获得特征是否能识别 major correction 样本。

可以强调的点：

- 使用 AP、ROC-AUC、Lift@20%、Precision@20% 衡量排序信号质量。
- 重点是发现“排在前 20% 的样本是否更容易包含重大修正”。
- 跨数据集验证能减少只在 COCO 上调参的偶然性。

边界：

- 这些是离线 benchmark 和校准结果。
- 不应说已经在线上生产流量中完成 A/B 验证。

## 8. Shadow-only rollout 和安全保护

简历原句：

设计 boundary/shape learned signal 的 shadow-only rollout 方案，保证实验分数不影响默认 Layer 5 排序，并通过 default-field equality、ordering equality、leakage guard、drift monitoring 与 safe review packet 控制上线风险。

详细解释：

learned boundary/shape score 在 benchmark 中有潜力，但不能直接替换默认排序。因此项目把它作为 shadow_scores 附加到 review queue，不改变 priority_score、rank、priority_bucket 和默认排序。每次 shadow replay 都验证默认字段是否完全一致、排序是否一致、是否读取了生产不可用 label/GT 字段。

可以强调的点：

- 这是工程上很重要的 rollout guardrail。
- 支持多窗口 shadow observation、drift monitoring、top-k Jaccard、missing feature analysis 和 safe review packet。
- 遇到 human usefulness 不足或 drift alert 时，决策是 shadow hold，而不是强行 promotion。

边界：

- shadow-only 表示实验信号只观察，不影响默认生产排序。
- 默认 promotion 需要额外的人审证据、SLO、回滚策略和独立验证。

## 9. Balanced human review pilots

简历原句：

组织 COCO 与 LVIS balanced human review pilots，并用 GT-derived IoU/Dice 构建 size-stratified correction effort summary；结果显示整体 learned signal 跨数据集不稳定，但 COCO/LVIS 小目标是明确高风险 slice，Major/Redo 分别达到 57.7%/51.7%，因此保持 shadow-only 并转向 size-aware calibration。

详细解释：

pilot 设计了四组，用于比较 learned boundary/shape score 和 current priority 的差异。COCO round 使用 60 个任务，每组 15 个；LVIS round2 使用 80 个任务，每组 20 个。任务里包含 GT reference、MobileSAM preview、editable mask 和 required review_outcome。COCO 聚合结果显示 high_learned_low_current 的 Major/Redo rate 为 46.7%，top_learned 为 20.0%，高于 control_current_top 的 6.7%；LVIS round2 中四组收敛在 5.0%-15.0%，说明 learned score 的整体 lift 不是跨数据集稳定成立。进一步用 GT-derived IoU/Dice 在 COCO/LVIS/DIS5K/CAMO/COD10K 上重算 correction effort，并按 GT mask area 分 small/medium/large 后，发现 COCO small-object Major/Redo rate 为 57.7%，LVIS small-object 为 51.7%，显著高于各自 large-object 的 22.2% 和 17.1%。新的结论是“整体排序信号需要继续校准，但小目标是明确、可行动的高风险 slice”。

可以强调的点：

- 这是一个设计过对照组的人工校准实验，不是随便挑几个样本看效果。
- COCO 结果支持 learned boundary/shape signal 存在早期探索价值。
- LVIS 独立验证暴露了整体 lift 的跨数据集稳定性问题，但 GT-derived 分层分析进一步定位到小目标这一明确高风险 slice。
- 项目没有因为单轮小样本结果好就直接改默认排序，而是把方向收敛到 size-aware risk calibration，体现保守上线意识。

边界：

- 两轮 pilot 总量仍小，且 GT-assisted，不等同生产真实复核。
- 当前结论是“整体 learned signal 跨数据集不稳定，但小目标 correction effort 明确更高”，不是“learned signal 已经整体优于 current priority”。
- 可以说“early calibration showed dataset-dependent signal and a clear small-object high-risk slice”，不要说“human validation conclusively proved production lift”。

## 面试回答模板

如果面试官问“这个项目最难的地方是什么”，可以回答：

最难的不是调用 MobileSAM，而是把模型预标注接入真实标注流之后，如何判断哪些 mask 需要人修、如何把人工修正变成可学习的数据、以及如何保证实验分数不会污染默认排序。我做了 mask quality metadata、prompt-stability uncertainty、correction delta、risk predictor 和 active-review queue，并且把 learned boundary/shape score 放在 shadow-only rollout 中，用 default-field equality、ordering equality 和 leakage guard 控制风险。

如果面试官问“这个项目有什么结果”，可以回答：

离线 benchmark 覆盖了 COCO、LVIS、DIS5K、CAMO、COD10K，观察到 boundary/shape 类预测时特征对 major correction discovery 有信号；但人工校准结论更细。COCO 60 任务 pilot 中 high learned/low current 的 Major/Redo rate 达到 46.7%，明显高于 current control 的 6.7%；LVIS 80 任务 round2 中整体 lift 没有稳定复现。为了避免只看整体均值，我又用 GT-derived IoU/Dice 重算 correction effort 并做 size-stratified analysis，发现 COCO 小目标 Major/Redo rate 为 57.7%，LVIS 小目标为 51.7%，都明显高于大目标。我的结论是：learned signal 作为整体排序策略还需要 shadow-only 校准，但小目标已经是明确高风险 slice，后续应做 size-aware risk calibration。

如果面试官问“为什么不用 SAM2 或 fine-tune”，可以回答：

这个项目的重点是标注闭环和风险排序，不是追求最大模型。MobileSAM 更轻量，适合本地 Docker 验证和快速 smoke test。SAM2 在项目里被明确放在 future work，避免把未验证 runtime 写进当前能力。

## 证据路径

- 总体 README: `README.md`
- 简历摘要旧版: `docs/resume_label_platform_summary.md`
- Label Studio 分割配置: `label_configs/image_segmentation.xml`
- MobileSAM/分割质量: `services/ml-backend/segmentation_quality.py`
- Prompt stability: `services/ml-backend/segmentation_uncertainty.py`
- Correction risk: `services/trainer/segmentation_correction_risk.py`
- Review queue: `services/trainer/segmentation_review_queue.py`
- Benchmark 模块: `image_segmentation/benchmark/`
- Shadow rollout 文档: `docs/segmentation_shadow_scoring_rollout.md`
- Balanced pilot closure: `demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/balanced_pilot_closure_report.md`
- COCO pilot review outcome: `demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_2026_06_04/label_studio_review_summary/coco_correction_effort_from_csv/review_group_outcome_summary.md`
- LVIS round2 review outcome: `demo_data/model_state/image_segmentation/benchmark/segmentation_balanced_pilot_round2_2026_06_07/correction_effort_from_project_11_export/review_group_outcome_summary.md`
- GT-derived size-stratified correction effort: `demo_data/model_state/image_segmentation/benchmark/gt_derived_correction_effort_summary_2026_06_07/gt_derived_correction_effort_summary.md`
