# 端到端测试运行指南

本文档记录算子集成完成后，如何运行端到端（压缩模型 + CUTLASS kernel）精度与速度测试。

## 1. 关键约定

- 仓库根目录：`/data/home/scxj523/run/wja/project/my/fake/`
- 默认避开节点：`--exclude=wqd10nah09g4`
- SLURM 输出目录：`out/`、`err/`（不存在需先 `mkdir -p out err`）
- CUTLASS 源码路径：`CUTLASS_ROOT`，默认 `/data/home/scxj523/run/wja/cutlass`，kernel 通过 JIT 编译
- conda 环境：`wja-cospaq`，CUDA 12.8（脚本内部已 `module load` / `conda activate`）

## 2. 脚本行为确认

经查源码：

| 脚本 | kernel 路径 | 说明 |
|---|---|---|
| `scripts/slurm/eval_compressed_accuracy.sh` | **不调用 CUTLASS kernel** | 精度评估走 fake-quant / pruned dequant，验证压缩本身的精度影响 |
| `scripts/slurm/bench_compressed_speed.sh` | 由 `KERNEL` 环境变量控制 | `KERNEL=0`（默认）→ dequant fallback；`KERNEL=1` → 加 `--kernel` 真正调用 CUTLASS 算子 |

**结论：速度测试必须显式传 `KERNEL=1`，否则跑的不是新集成的 kernel。**

相关位置：
- `scripts/slurm/bench_compressed_speed.sh`（`KERNEL_FLAG` 逻辑、`CUTLASS_ROOT` 导出）
- `scripts/bench_maxvit_dense_speed.py:33`、`scripts/bench_dinov3_vit7b16_dense_speed.py:31`（`--kernel` 参数）

## 3. 支持矩阵

- 模型：`maxvit`（variants：`tiny / small / base / large`）、`dinov3_vit7b16`
- 方法：`nvfp4`、`unstructured_sparse`、`semi_structured_sparse`、`nvfp4_unstructured_sparse`、`nvfp4_semi_structured_sparse`

## 4. 运行步骤

### Step 0：预检（每次新环境做一次）

```shell
ls /data/home/scxj523/run/wja/cutlass    # CUTLASS_ROOT 存在
mkdir -p out err
```

### Step 1：生成压缩 checkpoint

以 MaxViT tiny 为例，一次性生成 5 种方法的产物：

```shell
MODEL=maxvit MAXVIT_VARIANT=tiny \
  METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
  sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh
```

验证：

```shell
ls artifacts/checkpoints/maxvit_tiny/*/model.pt
```

DINOv3 同理：

```shell
MODEL=dinov3_vit7b16 \
  METHODS="nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse" \
  sbatch --exclude=wqd10nah09g4 scripts/slurm/prepare_compressed_models.sh
```

### Step 2：Kernel smoke 验证（建议在大规模评估前先跑一个 case）

```shell
MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=nvfp4 KERNEL=1 \
  sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
```

完成后检查：

```shell
tail -n 50 out/bench_compress_<jobid>.out          # 无 "fallback to dequant" 等异常
cat artifacts/results/maxvit_tiny_compressed/speed.csv
```

### Step 3：端到端精度全矩阵

```shell
for M in nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse; do
  MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=$M \
    sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
done
```

其他 variant：把 `MAXVIT_VARIANT=tiny` 替换为 `small / base / large`；`large` 自动用 `BATCH_SIZE=16`，其余 128。

DINOv3：

```shell
for M in nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse; do
  MODEL=dinov3_vit7b16 METHOD=$M \
    sbatch --exclude=wqd10nah09g4 scripts/slurm/eval_compressed_accuracy.sh
done
```

### Step 4：端到端速度全矩阵（带 CUTLASS kernel）

**注意：必须带 `KERNEL=1`。**

```shell
for M in nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse; do
  MODEL=maxvit MAXVIT_VARIANT=tiny METHOD=$M KERNEL=1 \
    sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
done
```

DINOv3：

```shell
for M in nvfp4 unstructured_sparse semi_structured_sparse nvfp4_unstructured_sparse nvfp4_semi_structured_sparse; do
  MODEL=dinov3_vit7b16 METHOD=$M KERNEL=1 \
    sbatch --exclude=wqd10nah09g4 scripts/slurm/bench_compressed_speed.sh
done
```

## 5. 结果路径

- MaxViT 精度：`artifacts/results/maxvit_<variant>_compressed/accuracy.csv`
- MaxViT 速度：`artifacts/results/maxvit_<variant>_compressed/speed.csv`
- DINOv3 精度：`artifacts/results/dinov3_vit7b16_compressed/accuracy.csv`
- DINOv3 速度：`artifacts/results/dinov3_vit7b16_compressed/speed.csv`
