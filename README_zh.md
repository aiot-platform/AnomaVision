# AnomaVision 中文使用指南

> 面向"标准件训练 + 输出 y/n"的生产场景，从零开始：部署 → 准备数据 → 训练 → 标定阈值（拿到分数）→ 检测 → 模型验证。

## 完整流程概览

```
1. 部署环境      uv venv + uv sync --extra cpu
2. 准备数据      dataset/<类别>/train/good/ 放入正常（合格）图片
3. 训练          anomavision train --config config.yml
4. 标定阈值      calibrate_threshold.py   ← 用正常图计算判定标准（关键步骤）
5. 检测          anomavision detect --config config.yml   → 每张图输出 y/n
6. 验证           eval（指标评估）/ validate（部署校验，可选）
```

---

## 1. 环境部署

要求：Windows / Linux，Python 3.10 ~ 3.12。

### 1.1 安装 uv（依赖管理工具）

```powershell
pip install uv
```

### 1.2 创建虚拟环境并安装依赖

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv sync --extra cpu
```

- 纯 CPU 机器用 `--extra cpu`（PyTorch CPU 版，约 1~2 GB 下载）。
- 有 NVIDIA GPU 用 `--extra cu124`（CUDA 12.4）。
- 以后每次换机器或装新依赖，重复 `uv sync --extra cpu` 即可，uv 会按 `pyproject.toml` + 锁文件自动对齐版本。

> Windows 上激活报"禁止运行脚本"时，先执行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，再激活。

### 1.3 验证安装

```powershell
uv run anomavision --version
```

---

## 2. 准备数据

只训练**正常（合格）标准件**图片。目录结构必须是 MVTec 风格：

```
dataset/
└── bottle/                  # 类别名，可改成你的产品名，如 cap、screw
    └── train/
        └── good/
            000.png
            001.png
            ...
```

> `class_name` 必须等于 `dataset_path` 下的这个子文件夹名。

在 `config.yml` 中确认三项：

```yaml
dataset_path: ./dataset          # 数据集根目录
class_name: "bottle"             # 与文件夹名一致
img_path: "./dataset/bottle/train/good"   # 标定阈值用的正常图目录
```

---

## 3. 训练

```powershell
uv run anomavision train --config config.yml
```

- 输出日志在 `logs/anomavision_<日期>_<时间>.log`。
- 模型保存到：`distributions/<算法>/<类别>/<run_name>/model.pt`
- **每次 train 都是全量重建**（不是增量）：往 `train/good` 里加新图后重新 train 即可，旧模型会被覆盖。
- 输出开头那两行 `Checking branch...` 是 git 检查，属正常现象，不影响训练。
- PatchCore 相关超参（coreset_ratio、max_memory_patches 等）都在 `config.yml` 中。

---

## 4. 标定阈值（拿到分数）★ 关键步骤

模型输出的**异常分数**越高代表越可能缺陷。判定规则：

```
分数 >= 阈值  →  n（不合格/异常）
分数 <  阈值  →  y（正常/合格）
```

`config.yml` 里默认的 `thresh_patchcore: 0.25` 只是仓库猜测值，**直接用它会导致全部图片判 n**。必须用你自己的正常图重新标定：

```powershell
uv run python scripts\calibrate_threshold.py --config config.yml
```

脚本做两件事：

1. 把模型跑在你指定的正常图上（默认 `config.img_path`），统计每张图的分数，打印：

```
Normal images evaluated: 209
Score min/mean/max: 0.3565 / 0.3837 / 0.3963
Recommended threshold (max + margin 0.05): 0.4463
```

2. 自动把结果写回 `config.yml` 的 `thresh_patchcore`（即 0.4463）。

### 阈值怎么来的（为什么是 0.4463）

```
0.4463 = 0.3963（正常图里最大的异常分数）+ 0.05（安全余量）
```

含义：只要新图分数不超过"正常图的最大分数 + 余量"，就判 y；超出才判 n。这样所有已知正常图都正确判 y。

### 余量取舍（--margin）

```powershell
uv run python scripts\calibrate_threshold.py --config config.yml --margin 0.1
```

- 调大：更保守，误报少，但轻微缺陷易漏检
- 调小：更灵敏，但易误报

> **重要：换数据集 / 重新训练后，分数分布会变，必须重新跑一次标定。**

---

## 5. 检测（输出 y/n）

```powershell
uv run anomavision detect --config config.yml
```

默认检测 `config.img_path` 指向的图片；也可临时指定别的图：

```powershell
uv run anomavision detect --config config.yml --img_path D:\test_images\某张图.png
# 或指向一个文件夹
uv run anomavision detect --config config.yml --img_path D:\test_images
```

输出格式（每张图一行）：

```
PER-IMAGE RESULT (y = normal, n = anomaly)
000.png: y
001.png: n
...
```

- 检测的是**训练过的正常图**时，应全部输出 y；出现 n 说明有异常。
- config 中 `enable_visualization / save_visualizations` 已设为 `false`，不再生成热力图等图片。

---

## 6. 模型验证

### 6.1 eval：指标评估（需要缺陷样本）

需要 MVTec 结构（含正常和缺陷测试图）：

```
dataset/bottle/
├── train/good/          # 训练用（正常）
└── test/
    ├── good/            # 测试（正常）
    └── broken/          # 测试（缺陷，类别名任意）
```

```powershell
uv run anomavision eval --config config.yml --dataset_path ./dataset --class_name bottle
```

会输出 AUC、精确率/召回率等指标，并自动寻找最优阈值。

> 若你只有正常图、没有缺陷样本，则跳过 eval，直接用第 4 步标定的阈值 + 第 5 步 detect。

### 6.2 validate：部署校验（可选，用于导出后的模型）

先导出（可选）：

```powershell
uv run anomavision export --config config.yml --format onnx
```

再校验模型文件完整性、推理性能、与参考模型输出一致性：

```powershell
uv run anomavision validate --model distributions\patchcore\bottle\anomav_exp\model.onnx --config config.yml --runs 20
```

---

## 7. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 检测正常图全输出 n | 阈值低于正常分数（如默认 0.25） | 重跑 `calibrate_threshold.py` 标定阈值 |
| 检测正常图全输出 y，缺陷也漏 | 余量过大 / 模型没学到该缺陷 | 调小 `--margin`，或增加正常图多样性 |
| `anomavision: error: unrecognized arguments: .yml` | 命令里 `config` 和 `.yml` 之间有空格 | 写成 `--config config.yml`（不要加空格） |
| train 命令闪退、无日志 | config.yml 语法错误 | 检查 YAML 缩进，可用 `uv run python -c "from anomavision.config import load_config; load_config('config.yml')"` 验证 |
| 模型训练时 CUDA 相关报错 | PyTorch 版本与硬件不匹配 | 改用 `--extra cpu` 重装 |

---

## 8. 常用命令速查

```powershell
# 训练
uv run anomavision train --config config.yml

# 标定阈值（每次重训后必跑）
uv run python scripts\calibrate_threshold.py --config config.yml

# 检测
uv run anomavision detect --config config.yml --img_path <图片或文件夹>

# 评估
uv run anomavision eval --config config.yml --dataset_path ./dataset --class_name <类别>

# 导出 ONNX
uv run anomavision export --config config.yml --format onnx

# 部署校验
uv run anomavision validate --model <模型文件> --config config.yml
```

日志统一在 `logs/` 目录；模型在 `distributions/<算法>/<类别>/<run_name>/`。
