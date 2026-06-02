# Label Platform 图像分割说明

本中文 README 只保留图像分割相关内容，作为后续中文维护版本。英文版 `README.md` 保留不动，可作为完整项目说明参考。

本仓库的图像分割路径围绕 **Label Studio OSS**、Docker 化 ML backend、SAM/MobileSAM 兼容预标注、人工修正反馈和轻量级修正风险学习构建。MobileSAM 是当前已验证的 SAM 兼容 mask 预标注运行时；SAM2 在本仓库中仅作为未来兼容表述，尚未实现。

## 图像分割能力概览

- Label Studio BrushLabels/RLE mask 预标注。
- 默认 deterministic placeholder mask，便于无大模型依赖时跑通流程。
- 可选 MobileSAM 兼容 backend，用于真实 SAM 风格预标注。
- 分割质量元数据：mask 面积、prompt bbox、mask bbox、bbox IoU、RLE 长度、贴边行为和 review flags。
- 可选 prompt-stability uncertainty：扰动 bbox prompt，比较多次 mask 的稳定性。
- 人工修正差异数据集：比较模型 mask 与人工修正 mask。
- 数据充足时训练 scikit-learn correction-risk predictor。
- Active-review queue：结合修正风险、不确定性、质量标记、几何启发式和轻量多样性信号，对样本进行 review 优先级排序。

## Image Segmentation Benchmark v0.1

Benchmark v0.1 用于评估 Layer 1-5 分割反馈闭环是否能比随机 review 更好地发现可能需要重大修正的样本。MVP 仅支持 COCO val2017。它只下载 COCO `val2017` 图像和 `instances_val2017.json`，不会下载完整的 COCO 2017 train/test 数据集。

显式准备本地数据：

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset coco_val2017 \
  --output-dir data/external
```

数据会存储在：

```text
data/external/coco/
  val2017/
  annotations/instances_val2017.json
```

运行 preflight，不隐式下载任何内容：

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.coco.yaml
```

构建 manifest：

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.coco.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_manifest.jsonl
```

运行 benchmark：

```bash
python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1 \
  --backend mobile_sam_or_existing_backend \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk
```

输出：
- `predictions.jsonl`：模型预标注 mask，包含 Layer 1 质量/review 元数据和可选 Layer 2 不确定性。
- `correction_delta_dataset.jsonl`：模型与 COCO GT 的修正差异，标记为 `public_gt_simulated_human_annotation`。
- `review_queue.jsonl`：Layer 5 优先级队列，其中 delta 仅用于 `evaluation_only`。
- `evaluation_report.json` 和 `evaluation_report.md`：机器可读与人类可读指标。

报告指标：
- model/GT IoU、Dice、precision 和 recall 衡量预标注 mask 相对公开 ground truth 的质量。
- major correction rate 和 severity distribution 总结模拟人工修正的显著程度与出现频率。
- grouped dataset/tag metrics 展示小目标、贴边、细长或拥挤样本上的表现。
- precision@10/20% 和 recall@10/20% 衡量 review 队列顶部发现重大修正样本的能力。
- lift@10/20% 将完整 Layer 5 优先级排序与随机 baseline 的重大修正率进行比较。
- average precision 总结所有重大修正样本上的排序质量。

当前限制：
- COCO val2017 和 LVIS val 使用 COCO-style instance annotation；DIS5K、COD10K、CAMO 和 Open Images subset 支持本地 manifest/preflight loader。
- 此 MVP 中 COCO 没有显式的低对比度、遮挡或截断元数据。
- placeholder backend 是确定性 fallback；MobileSAM 需要已有 backend 依赖和 checkpoint。

### 附加数据集准备

列出支持的数据集：

```bash
python -m image_segmentation.benchmark.list_datasets
```

通用原则：`download_data` 必须由用户显式调用。`build_manifest` 和 `run_benchmark` 不会隐式下载数据。手动数据集需要用户自行确认许可条款和下载来源。`data/external/` 已被 git 忽略，不应提交。Ground truth 和 correction delta 只用于 benchmark/evaluation，不能进入 priority scoring。Natural validation 不应根据模型失败样本筛选。

LVIS val 用于长尾类别、小目标和多实例复杂场景。annotation 下载到 `data/external/lvis/annotations/lvis_v1_val.json`，图片复用 `data/external/coco/val2017`。

```bash
python -m image_segmentation.benchmark.download_data --dataset lvis_val --output-dir data/external --dry-run
python -m image_segmentation.benchmark.download_data --dataset lvis_val --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.lvis500.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl
```

DIS5K 用于高精细边界和复杂前景轮廓。第一版采用手动放置：

```text
data/external/dis5k/
  images/
  masks/
```

```bash
python -m image_segmentation.benchmark.download_data --dataset dis5k --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.dis5k300.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl
```

COD10K 和 CAMO 用于伪装、低对比度目标，以及 prompt-stability uncertainty 压力测试。它们复用 mask-folder loader，并默认添加 `low_contrast` 和 `camouflaged_object` 标签。期望手动放置：

```text
data/external/cod10k/images/
data/external/cod10k/masks/
data/external/camo/images/
data/external/camo/masks/
```

```bash
python -m image_segmentation.benchmark.download_data --dataset cod10k --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.cod10k300.yaml
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.cod10k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_cod10k300_manifest.jsonl

python -m image_segmentation.benchmark.download_data --dataset camo --output-dir data/external
python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.camo300.yaml
```

Open Images V7 segmentations 用于遮挡、截断、group object 和 instance segmentation 多样性。第一版建议只下载 validation subset，不建议全量下载。可选 FiftyOne 路径需要 `pip install fiftyone`，并导出为：

```text
data/external/open_images_v7/validation/
  images/
  segmentations/
  metadata.json
  annotations.jsonl
```

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset open_images_v7_segmentations \
  --output-dir data/external \
  --split validation \
  --max-samples 1000 \
  --use-fiftyone true

python -m image_segmentation.benchmark.preflight --config configs/benchmark_v0_1.open_images500.yaml
```

许可和来源说明：COCO 遵循 COCO 图片/annotation 条款。LVIS annotation 为 CC BY 4.0，图片遵循其来源条款。DIS5K、COD10K、CAMO 和 Open Images 使用前都需要确认当前数据集许可与 image-level terms。

### LVIS500 validation

LVIS500 用于验证 COCO1000 上的结论能否迁移到更长尾、更拥挤、小目标更多的 instance segmentation 数据集。COCO1000 已证明 MobileSAM benchmark、correction delta、OOF correction-risk 信号和 Layer 5 review prioritization 初步可用；LVIS 是修改默认权重前的下一步压力测试。在 LVIS 结果出来前，不要做 Layer 6，也不要改默认 Layer 5 权重。

准备 LVIS val annotation。LVIS 使用已有 COCO val2017 图片目录 `data/external/coco/val2017`；下面命令只下载 annotation。

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset lvis_val \
  --output-dir data/external
```

运行 preflight 并构建 LVIS500 manifest：

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --json-output demo_data/model_state/image_segmentation/benchmark/lvis500_preflight.json

python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.lvis500.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest_summary.json
```

使用 MobileSAM GPU 跑 benchmark，不开启 prompt-stability uncertainty：

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

python -m image_segmentation.benchmark.preflight_backend \
  --backend mobile_sam \
  --checkpoint /home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt \
  --device cuda

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

运行 OOF risk evaluation、weight ablation 和 COCO-vs-LVIS 对比：

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000

python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42

python -m image_segmentation.benchmark.compare_benchmark_runs \
  --left-name COCO1000 \
  --left-report demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu/evaluation_report.json \
  --left-oof demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof/oof_summary.json \
  --left-ablation demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --right-name LVIS500 \
  --right-report demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/evaluation_report.json \
  --right-oof demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_summary.json \
  --right-ablation demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --output demo_data/model_state/image_segmentation/benchmark/coco1000_vs_lvis500_comparison.md
```

解释方式：rare category correction rate 用于判断 LVIS 长尾类别是否更难，但除非预测时也有同等 category metadata，否则不要把 GT-only frequency metadata 用进生产 scoring。如果 `risk_heavy` 在 COCO1000 和 LVIS500 上都优于 current，默认权重仍保持不变，同时把 `risk_heavy` 标记为强 experimental candidate。如果 LVIS 与 COCO 结论不一致，先继续验证 DIS5K 或 COD10K，再考虑默认权重调整。

## 启动服务栈

```bash
cp .env.example .env
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
docker compose --env-file .env -f infra/docker-compose.yml ps
```

本地 URL：
- Label Studio OSS: `http://localhost:18080`
- ML backend health: `http://localhost:9090/health`
- Trainer health: `http://localhost:9091/health`

## Label Studio 图像分割项目

图像分割使用单个前景标签 `Object`，并保持 Label Studio -> prediction -> human correction -> webhook -> training data -> retrain -> new pre-label 这条契约可测试。

Label config：
- `label_configs/image_segmentation.xml`

Bootstrap/import 一个 segmentation review project：

```bash
export LABEL_STUDIO_URL=http://localhost:18080
export LABEL_STUDIO_API_TOKEN='<your-token>'
scripts/bootstrap_label_studio_image_segmentation_review.py
scripts/import_image_segmentation_review_tasks_to_label_studio.py
```

如果图像任务使用 `/data/local-files/?d=...`，需要在 Label Studio 中配置 source storage：
- `Project Settings` -> `Cloud Storage` -> `Add Source Storage` -> `Local Files`
- Absolute local path: `/label-studio/files/images`
- 启用 `Treat every bucket object as a source file`

连接 ML backend：
- `Project Settings` -> `Model` -> `Connect Model`
- URL: `http://ml-backend:9090`

本地图像路径能够解析的原因：
- compose mount: `demo_data/local-files` -> `/label-studio/files`
- Label Studio document root: `/label-studio/files`
- task URL `d=images/demo_blue.png` 会解析到 `/label-studio/files/images/demo_blue.png`

## 分割反馈 Pipeline

```text
Image task + bbox prompt
        ↓
Dockerized MobileSAM backend
        ↓
BrushLabels/RLE pre-label
        ↓
Layer 1: mask_quality + review metadata
        ↓
Layer 2: prompt-stability uncertainty, optional
        ↓
Human review/correction in Label Studio
        ↓
Layer 3: model-vs-human correction delta dataset
        ↓
Layer 4: correction-risk predictor
        ↓
Layer 5: active-review queue
```

Layer 1：分割质量元数据
- 向 BrushLabels/RLE predictions 添加 `mask_quality` 和 `review` 元数据。
- 捕获 mask area、prompt bbox、mask bbox、bbox IoU、RLE length、border touching 和 review flags。

Layer 2：Prompt-stability uncertainty
- 可选，默认禁用。
- 扰动 bbox prompts，对多个变体运行 SAM 兼容 backend，计算 pairwise mask IoU 和 disagreement area，并存储 `uncertainty` 元数据。

Layer 3：人工修正差异数据集
- 比较模型生成 masks 与人工修正的 Label Studio annotations。
- 计算 IoU、Dice、added/removed area、correction area ratio、bbox alignment、centroid shift 和 correction severity。
- 持久化 `correction_delta_dataset.jsonl`，并在 segmentation metadata 中汇总 correction deltas。

Layer 4：学习式修正风险预测器
- 使用 scikit-learn `LogisticRegression`。
- 仅在存在足够 correction delta records 时训练。
- 使用来自 `mask_quality`、`review`、`uncertainty` 和 geometry metadata 的预修正特征。
- 预测某个 pre-label 是否可能需要重大人工修正。
- 不使用 `delta` metrics 作为输入特征，以避免 target leakage。

Layer 5：Active-review 队列
- 结合 correction-risk predictions、prompt-stability uncertainty、mask quality review flags、geometry heuristics 和轻量 diversity signal。
- 生成排序后的 JSONL records，包含 score components、review reasons、source metadata 和 evaluation-only deltas。
- 不修改 Label Studio tasks，不从 unlabeled pool 采样，也不基于队列重新训练模型。

这是 active-learning-style review prioritization，不是完整的 active learning acquisition loop。

## 运行时预标注 Backend

默认 placeholder backend 无需大模型依赖：

```bash
export IMAGE_SEG_BACKEND=placeholder
```

MobileSAM 兼容路径：

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_MODEL_TYPE=vit_t
export IMAGE_SEG_CHECKPOINT=/model-state/image_segmentation/mobile_sam.pt
export IMAGE_SEG_DEVICE=cpu
```

可选 prompt-stability uncertainty：

```bash
export IMAGE_SEG_PROMPT_STABILITY_ENABLED=true
export IMAGE_SEG_PROMPT_STABILITY_VARIANTS=7
export IMAGE_SEG_PROMPT_STABILITY_JITTER=0.05
```

SAM2 未来兼容配置，目前未实现 runtime：

```bash
export IMAGE_SEG_BACKEND=sam2
export IMAGE_SEG_MODEL_ID=facebook/sam2-hiera-large
export IMAGE_SEG_DEVICE=cuda
```

当 `IMAGE_SEG_BACKEND` 为 `mobilesam`、`sam` 或 `sam2` 时，ML backend 预期可选模型库已安装在 runtime image 中。如果请求真实 backend 但不可用，prediction 会 fallback 到 placeholder mask，并在 prediction confidence metadata 中包含 `requested_backend`、`backend_error` 和 `fallback`。

SAM 兼容 backend 会按如下顺序选择一个 box prompt：
- `task.data.bbox` / `task.data.box`
- `task.meta.bbox` / `task.meta.box`
- `data.candidates[]` 或 `meta.candidates[]` 中最佳有效 bbox
- predictions 或 annotations 中第一个有效 Label Studio `rectanglelabels` result
- 中心框 fallback

带 `score` 或 `confidence` 的 candidate boxes 会使用最高分有效 box；无分数 candidates 保持输入顺序。List 值为像素 `x_min, y_min, x_max, y_max`，mapping 值可使用 `x/y/width/height` 或 `x_min/y_min/x_max/y_max`。百分比 box 必须将 `unit`、`units` 或 `coordinate_system` 设为 `percent`、`percentage`、`pct` 或 `%`；`normalized=true` 的 box 使用 0-1 坐标；Label Studio `rectanglelabels` 使用百分比 `x/y/width/height`。Result `meta` 会记录 `prompt`、像素 `prompt_box` 和 `mask_bbox`，用于可追踪性。

## 产物位置

- `demo_data/model_state/image_segmentation/metadata.json`
- `demo_data/model_state/image_segmentation/placeholder_model.json`
- `demo_data/model_state/image_segmentation/last_training_dataset.jsonl`
- `demo_data/model_state/image_segmentation/correction_delta_dataset.jsonl`
- `demo_data/model_state/image_segmentation/correction_risk/classifier.joblib`
- `demo_data/model_state/image_segmentation/correction_risk/metadata.json`
- `demo_data/model_state/image_segmentation/correction_risk/feature_names.json`
- `demo_data/model_state/image_segmentation/correction_risk/training_dataset.jsonl`
- `demo_data/model_state/image_segmentation/review_queue.jsonl`

Segmentation metadata endpoints：
- `http://localhost:9091/models/image-segmentation/current`
- `http://localhost:9090/models/image-segmentation/current`

## Docker GPU MobileSAM Runtime

可选 GPU runtime：

```bash
# prerequisite: Docker can access the GPU
docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime nvidia-smi

# prerequisite: this checkpoint exists locally
# models/mobilesam/mobile_sam.pt
cd infra
docker compose build ml-backend-gpu
docker compose up ml-backend-gpu
```

GPU service 默认将 host `9092` 映射到 container `9090`；设置 `ML_BACKEND_GPU_PORT` 可覆盖 host port。它使用 `MODEL_STATE_PATH=/app/demo_data/model_state/current_image_segmentation_model.json`、`IMAGE_SEG_BACKEND=mobilesam`、`IMAGE_SEG_MODEL_TYPE=vit_t`、`IMAGE_SEG_CHECKPOINT=/app/models/mobilesam/mobile_sam.pt` 和 `IMAGE_SEG_DEVICE=cuda` 运行。

checkpoint 从 `../models:/app/models` 挂载；不要将 model weights 或 checkpoints 提交到 git。要将 Label Studio 连接到 GPU backend，请使用 `http://ml-backend-gpu:9090` 作为 ML backend URL。真实 MobileSAM pre-labels 会包含 `model_version=mobilesam-seg-v0001`、`prediction_source=mobilesam-image-segmentation`、`backend=mobilesam`、prompt metadata，以及长度大于零的 Brush RLE；不应包含 placeholder fallback metadata。

如果 WSL 上的 Docker Desktop 在发布 host ports `9091` 或 `9092` 时报告 port-forwarding error，请保持 container URL 不变，只覆盖本地检查用的 host ports，例如：

```bash
TRAINER_PORT=19091 ML_BACKEND_GPU_PORT=19092 docker compose up -d trainer ml-backend-gpu
```

Label Studio 与服务仍会通过 Docker network 使用 `trainer:9091` 和 `ml-backend-gpu:9090` 通信。

## Docker MobileSAM Smoke Test

在 segmentation review project 已导入 tasks 且已有 MobileSAM prediction 后运行 smoke test。新的 Label Studio 数据库可能会分配不同 project ID，因此请使用 bootstrap/import 脚本打印的 project ID，不要硬编码 project `3`。

host health URL 通常是 `http://127.0.0.1:9092`，而 Label Studio 必须使用 Docker-network backend URL `http://ml-backend-gpu:9090`。

```bash
python scripts/smoke_mobilesam_segmentation_docker.py \
  --compose-dir infra \
  --project-id <segmentation-project-id> \
  --expected-backend-url http://ml-backend-gpu:9090 \
  --expected-model-version mobilesam-seg-v0001 \
  --host-ml-backend-url http://127.0.0.1:9092 \
  --out-dir /tmp
```

如果 host `9092` 没有在 WSL 中转发，请使用：

```bash
python scripts/smoke_mobilesam_segmentation_docker.py \
  --compose-dir infra \
  --project-id <segmentation-project-id> \
  --expected-backend-url http://ml-backend-gpu:9090 \
  --expected-model-version mobilesam-seg-v0001 \
  --host-ml-backend-url http://127.0.0.1:19092 \
  --out-dir /tmp
```

smoke test 会检查 Docker/compose、container health、checkpoint/model state、Label Studio DB state、最新 prediction metadata（包括 `mask_quality`、`review`、`needs_review`、`review_priority` 和 `review_reason`），并将解码后的 mask 和 overlay PNG 写入 `/tmp`；失败会按 check 报告。

Prompt-stability uncertainty 在 smoke script 中是可选的。先以 `IMAGE_SEG_PROMPT_STABILITY_ENABLED=true` 启动 backend，生成新的 prediction，然后添加 `--enable-prompt-stability`，要求最新持久化 prediction 中存在 `uncertainty` metadata。默认 smoke 命令不要求该项，因为 prompt-stability 会执行多次 MobileSAM inference。

## DIS5K300 边界压力验证

### 验证目的

DIS5K300 用于 foreground segmentation / 高精细边界压力测试。它不是长尾类别验证。它主要测试精细边界质量、`correction_area_ratio`、细长结构、高边界复杂度，以及 bbox prompt 看似正确但 mask 边界仍需大量修正的情况。

DIS5K 是 manual dataset。CLI 只输出人工准备说明，不隐式下载大数据，也不写死不稳定的 Google Drive/Baidu 链接：

```bash
python -m image_segmentation.benchmark.download_data \
  --dataset dis5k \
  --output-dir data/external \
  --dry-run
```

### 数据目录

默认本地目录结构：

```text
data/external/dis5k/
  images/
  masks/
```

image 和 mask 通过文件名 stem 配对。mask 应是二值 mask，或可用 `foreground > 0` 转成二值。默认要求 image/mask 尺寸一致。不自动下载 DIS5K；不自动 resize GT mask，除非 config 明确允许。

### Preflight 命令

```bash
python -m image_segmentation.benchmark.preflight \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --json-output demo_data/model_state/image_segmentation/benchmark/dis5k300_preflight.json
```

preflight 会检查 `images_dir` / `masks_dir`、配对数量、missing image/mask count、empty masks、image/mask size mismatch、合法 bbox / area 和 `sample_check`。

### Build Manifest 命令

```bash
python -m image_segmentation.benchmark.build_manifest \
  --config configs/benchmark_v0_1.dis5k300.yaml \
  --output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl \
  --summary-output demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest_summary.json
```

manifest 包含 `mask_path`、`category_name=foreground_object`、`gt_bbox_xyxy`、`difficulty_tags` 和 `boundary_metadata`。`boundary_metadata` 来自 GT mask，只能用于诊断和 evaluation breakdown；不能用于生产排序和 risk features，除非未来实现等价的 prediction-time boundary features。

### MobileSAM GPU Benchmark 命令

```bash
export IMAGE_SEG_BACKEND=mobilesam
export IMAGE_SEG_CHECKPOINT=/home/yuanli/projects/label-platform/models/mobilesam/mobile_sam.pt
export IMAGE_SEG_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0

python -m image_segmentation.benchmark.run_benchmark \
  --manifest demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_manifest.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu \
  --backend mobile_sam \
  --enable-prompt-stability false \
  --risk-model-dir demo_data/model_state/image_segmentation/correction_risk \
  --resume true
```

完整 DIS5K300 不开启 prompt-stability。使用 GPU，不允许 fallback。输出包括 `predictions.jsonl`、`correction_delta_dataset.jsonl`、`review_queue.jsonl`、`evaluation_report.json`、`evaluation_report.md` 和 `runtime_metadata.json`。

### 边界指标解释

- boundary IoU：带容忍边界带之间的 overlap。
- boundary F1：boundary precision 和 boundary recall 的调和平均。
- boundary precision：model boundary 被 GT boundary band 匹配的比例。
- boundary recall：GT boundary 被 model boundary band 匹配的比例。
- boundary error area ratio：model/human mask 对称差面积除以图像面积。

IoU 高但 boundary F1 低，表示区域大体对但边界差。`high_boundary_complexity` / `thin_structure` 失败率高，说明可能需要 boundary-aware prediction-time features。

### Prediction-Time Boundary/Shape Features

benchmark 还会记录只来自 predicted mask 和 image size 的 prediction-time boundary/shape features：`pred_area_ratio`、`pred_bbox_area_ratio`、`pred_extent`、`pred_aspect_ratio`、`pred_touches_border`、`pred_boundary_complexity`、`pred_boundary_density`、`pred_component_count`、`pred_largest_component_ratio`、`pred_hole_count` 和 `pred_thinness_proxy`。

这些字段写入 `prediction_features` 和 `mask_quality.prediction_time_boundary_shape`。它们可以安全用于 Layer 5 priority 实验，因为不使用 GT mask、IoU/Dice、boundary delta、major correction label 或 human mask。

Evaluation-only 字段包括 `boundary_metadata`、`evaluation_only.delta`、IoU/Dice、boundary IoU/F1/precision/recall、correction delta、correction severity 和 `major_correction` label。它们可以用于离线诊断和评估 label，但不能进入生产 scoring 或 learned-fusion 输入特征。

运行 prediction-time feature diagnostics：

```bash
python -m image_segmentation.benchmark.diagnose_prediction_features \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/review_queue.jsonl \
  --output-dir /tmp/dis5k300_boundary_shape_diagnostics
```

输出包括 `prediction_feature_diagnostics.json`、`prediction_feature_diagnostics.md` 和 `prediction_feature_bins.jsonl`。报告会统计 missingness、分位数、单变量 AP/ROC-AUC/PR-AUC、Spearman rank correlation、分桶 major-correction rate，以及每个 feature 的方向建议。

### OOF 风险评估

```bash
python -m image_segmentation.benchmark.crossfit_risk_evaluation \
  --delta-dataset demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/correction_delta_dataset.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof \
  --n-splits 5 \
  --random-seed 42 \
  --bootstrap-iters 1000
```

初步有效性标准：

```text
OOF lift@20 > 1.5
OOF AP > base rate
OOF precision@20 > random expected precision
```

### 权重消融

```bash
python -m image_segmentation.benchmark.ablate_review_weights \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof/oof_review_queue.jsonl \
  --output-dir demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof_weight_ablation \
  --bootstrap-iters 1000 \
  --random-seed 42
```

重点比较 `current_full_priority`、`risk_heavy`、`risk_only` 和 `no_diversity`。

`boundary_shape_experimental` preset 会给 `boundary_shape_score` 非零权重，但默认 Layer 5 权重保持不变。`boundary_shape_rank_score` 和 `boundary_shape_calibrated_score` 分别比较 rank-normalized 与 robust log/clipped prediction-time boundary score。它们都只用于实验比较，除非跨数据集稳定，否则不要作为默认。

运行 leakage-safe OOF learned fusion：

```bash
python -m image_segmentation.benchmark.learn_boundary_shape_fusion \
  --review-queue demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/review_queue.jsonl \
  --output-dir /tmp/dis5k300_boundary_shape_learned_fusion \
  --n-splits 5 \
  --random-seed 42
```

learned fusion 会比较 `current_full_priority`、`risk_only`、`risk_heavy`、`boundary_shape_experimental`、`boundary_shape_rank_score`、`boundary_shape_calibrated_score`、`learned_boundary_shape_only` 和 `learned_current_plus_boundary_shape`。每个 fold 的 scaler/model 只在 train fold fit，validation fold 只 transform/predict；如果没有 scikit-learn，会使用 numpy logistic fallback。

### 三方比较

```bash
python -m image_segmentation.benchmark.compare_benchmark_runs \
  --run COCO1000:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_coco1000_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --run LVIS500:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_lvis500_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --run DIS5K300:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu/evaluation_report.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof/oof_summary.json:demo_data/model_state/image_segmentation/benchmark/benchmark_v0_1_dis5k300_mobile_sam_gpu_oof_weight_ablation/weight_ablation.json \
  --output demo_data/model_state/image_segmentation/benchmark/coco_lvis_dis5k_comparison.md
```

三方比较用于判断哪个数据集更难、`risk_heavy` 是否跨数据集优于 current、DIS5K 是否暴露边界型失败，以及是否需要新增 prediction-time boundary features。

### 可选 DIS5K500

只有在 DIS5K300 证据不足时才跑 DIS5K500：`positive_count < 30`、CI 太宽、ablation 结果不稳定，或 boundary diagnostics 样本不足。

配置模板：

```yaml
benchmark_id: benchmark_v0_1_dis5k500
random_seed: 42
max_samples_total: 500

datasets:
  dis5k:
    enabled: true
    images_dir: data/external/dis5k/images
    masks_dir: data/external/dis5k/masks
    max_samples: 500

sampling:
  strategy: stratified
  image_diversity: true
  include_tags:
    - small_object
    - touches_border
    - elongated_object
    - thin_structure
    - high_boundary_complexity
```

然后用 `dis5k500` 输出路径依次运行 `build_manifest`、`run_benchmark`、`crossfit_risk_evaluation` 和 `ablate_review_weights`。

### 禁止事项

- 不要用 GT-derived `boundary_metadata` 做生产排序。
- 不要把 boundary delta metrics 当作 risk features。
- 不要把 `boundary_shape_experimental` 当作默认权重修改。
- 不要仅凭 DIS5K 一次结果修改默认 Layer 5 权重。
- 不要做 Layer 6。
- 不要在完整 DIS5K300 上默认开启 prompt-stability，除非明确做 uncertainty sanity run。

## 重新训练与验证

触发并检查 segmentation retrain：

```bash
scripts/trigger_image_segmentation_retrain.sh
scripts/inspect_image_segmentation_metadata.sh
```

运行确定性 segmentation gate：

```bash
scripts/run_image_segmentation_validation_gate.sh
```

基础服务验证：

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --build
docker compose --env-file .env -f infra/docker-compose.yml ps
curl -sS http://localhost:9090/health
curl -sS http://localhost:9091/health
scripts/run_image_segmentation_validation_gate.sh
```

## 边界说明

- 这还不是生产级 active learning 平台。
- review 队列不会自动修改 Label Studio tasks 或重新训练模型。
- 不会 fine-tune MobileSAM。
- 未实现 SAM2。
- 不会自动判断标签正确性。
- 图像检测和文本 NER 仍是确定性规则 demo 路径，不属于本文档范围。

## 停止服务

```bash
docker compose --env-file .env -f infra/docker-compose.yml down
```
