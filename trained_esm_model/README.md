# 模型下载说明

本文件夹用于存放训练好的 ESM 模型文件。

## 如何下载模型

### 方法 1: 使用 Hugging Face Hub (推荐)

使用 `huggingface_hub` 库下载模型：

```bash
pip install huggingface_hub
```

然后运行以下 Python 代码：

```python
from huggingface_hub import snapshot_download

# 下载模型到当前文件夹
snapshot_download(
    repo_id="jackkuo/BCE-Vir-Prediction_model",
    local_dir="./",
    local_dir_use_symlinks=False
)
```

或者在命令行中使用 `huggingface-cli`：

```bash
huggingface-cli download jackkuo/BCE-Vir-Prediction_model --local-dir ./ --local-dir-use-symlinks False
```

### 方法 2: 使用 Git LFS

如果安装了 Git LFS，可以直接克隆：

```bash
git lfs install
git clone https://huggingface.co/jackkuo/BCE-Vir-Prediction_model .
```

### 方法 3: 手动下载

访问模型页面：https://huggingface.co/jackkuo/BCE-Vir-Prediction_model

在文件列表中选择需要的文件进行下载，并保存到本文件夹中。

## 模型文件结构

下载完成后，本文件夹应包含以下文件：
- `config.json` - 模型配置文件
- `model.safetensors` - 模型权重文件（使用 safetensors 格式）
- `tokenizer_config.json` - 分词器配置文件
- `vocab.txt` - 词汇表文件
- `special_tokens_map.json` - 特殊标记映射文件

## 使用说明

模型下载完成后，在代码中使用以下路径加载模型：

```python
model_path = "trained_esm_model"
model = EsmForSequenceClassification.from_pretrained(model_path)
tokenizer = EsmTokenizer.from_pretrained(model_path)
```
