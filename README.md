# BCE-Vir-Prediction

基于 ESM (Evolutionary Scale Modeling) 的病毒表位预测工具。本工具使用预训练的 ESM 分类模型对蛋白质序列进行滑动窗口预测，识别潜在的抗原表位和功能域。

## 功能特性

- **表位预测** (`bcepre_predict_logits.py`): 使用预训练的 ESM 分类模型，通过滑动窗口切分蛋白质序列，对每个子序列进行分类预测（如是否为抗原表位、功能域等），并保存预测结果和对应的 logits 值。
- **氨基酸概率预测** (`bcepre_predict_softmax.py`): 将滑动窗口预测结果转化为按氨基酸位置聚合的概率值，输出包含氨基酸类型、表位概率和覆盖次数的结果表格。

## 模型

预训练模型可从 Hugging Face 下载：

- **模型仓库**: [jackkuo/BCE-Vir-Prediction_model](https://huggingface.co/jackkuo/BCE-Vir-Prediction_model)

详细的模型下载说明请参考 [`trained_esm_model/README.md`](trained_esm_model/README.md)

## 安装依赖

使用 `requirements.txt` 安装所有依赖：

```bash
pip install -r requirements.txt
```

如果使用 Hugging Face Hub 下载模型，需要额外安装：

```bash
pip install huggingface_hub
```

## 使用方法

### 步骤 1: 下载模型

首先需要下载预训练模型到 `trained_esm_model` 文件夹。详细步骤请参考 [`trained_esm_model/README.md`](trained_esm_model/README.md)

### 步骤 2: 准备输入文件

将待预测的蛋白质序列文件（FASTA 格式）放置在 `example_data` 文件夹中，或修改脚本中的输入文件路径。

### 步骤 3: 运行表位预测

运行 `bcepre_predict_logits.py` 脚本进行表位预测：

```bash
python bcepre_predict_logits.py
```

该脚本会：
- 读取 FASTA 格式的蛋白质序列文件
- 使用滑动窗口（默认最小窗口大小为 5）切分序列
- 对每个子序列进行分类预测
- 输出包含以下字段的 CSV 文件：
  - `sequence`: 子序列
  - `window_size`: 窗口大小
  - `prediction`: 预测类别
  - `logit_0`, `logit_1`, ...: 各类别的 logits 值

输出文件默认保存在 `predictions/` 文件夹中。

### 步骤 4: 计算氨基酸位置概率

运行 `bcepre_predict_softmax.py` 脚本将预测结果转化为氨基酸位置的聚合概率：

```bash
python bcepre_predict_softmax.py
```

该脚本会：
- 读取由 `bcepre_predict_logits.py` 生成的 CSV 文件
- 计算每个子序列的表位概率（使用 softmax 函数）
- 按氨基酸位置聚合概率值
- 输出包含以下字段的 CSV 文件：
  - `position`: 氨基酸位置（从 1 开始）
  - `amino_acid`: 氨基酸类型
  - `probability`: 该位置的表位概率（所有覆盖该位置的窗口预测的平均值）
  - `coverage`: 覆盖该位置的窗口数量

## 项目结构

```
BCE-Vir-Prediction/
├── bcepre_predict_logits.py          # 表位预测脚本
├── bcepre_predict_softmax.py         # 氨基酸概率预测脚本
├── requirements.txt                   # Python 依赖包列表
├── trained_esm_model/                # 预训练模型文件夹（需要下载）
│   └── README.md                     # 模型下载说明
├── example_data/                     # 示例数据文件夹
│   ├── PDCoV_GDSG10_RBD_aa.fa       # 示例输入 FASTA 文件
│   └── PDCoV_GDSG10_RBD_aa_8aa_logits.csv  # 示例中间结果
├── predictions/                      # 预测结果输出文件夹（自动创建）
├── README.md                         # 本文件
├── LICENSE                           # 许可证文件
└── .gitignore                        # Git 忽略文件配置
```

## 参数配置

### bcepre_predict_logits.py

在脚本中可以调整以下参数：

- `model_path`: 模型路径（默认: `"trained_esm_model"`）
- `fasta_file`: 输入 FASTA 文件路径
- `min_window_size`: 最小滑动窗口大小（默认: 5）
- `batch_size`: 批处理大小（默认: 8）
- `output_file`: 输出 CSV 文件路径

### bcepre_predict_softmax.py

在脚本中可以调整以下参数：

- `input_csv`: 输入 CSV 文件路径（由 `bcepre_predict_logits.py` 生成）
- `output_csv`: 输出 CSV 文件路径

## 注意事项

1. 确保已下载模型文件到 `trained_esm_model` 文件夹
2. 根据可用 GPU 内存调整 `batch_size` 参数
3. 对于较长的序列，预测过程可能需要较长时间
4. 输出文件夹 `predictions/` 会在运行脚本时自动创建

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 引用

如果您使用本工具进行研究，请引用相关的模型和代码库。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
