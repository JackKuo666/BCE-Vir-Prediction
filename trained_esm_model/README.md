# Model Download Instructions

This folder is used to store the trained ESM model files.

## How to Download the Model

### Method 1: Using Hugging Face Hub (Recommended)

Use the `huggingface_hub` library to download the model:

```bash
pip install huggingface_hub
```

Then run the following Python code:

```python
from huggingface_hub import snapshot_download

# Download the model to the current folder
snapshot_download(
    repo_id="jackkuo/BCE-Vir-Prediction_model",
    local_dir="./",
    local_dir_use_symlinks=False
)
```

Or use `huggingface-cli` in the command line:

```bash
huggingface-cli download jackkuo/BCE-Vir-Prediction_model --local-dir ./ --local-dir-use-symlinks False
```

### Method 2: Using Git LFS

If Git LFS is installed, you can clone directly:

```bash
git lfs install
git clone https://huggingface.co/jackkuo/BCE-Vir-Prediction_model .
```

### Method 3: Manual Download

Visit the model page: https://huggingface.co/jackkuo/BCE-Vir-Prediction_model

Select the required files from the file list to download and save them to this folder.

## Model File Structure

After downloading, this folder should contain the following files:
- `config.json` - Model configuration file
- `model.safetensors` - Model weights file (in safetensors format)
- `tokenizer_config.json` - Tokenizer configuration file
- `vocab.txt` - Vocabulary file
- `special_tokens_map.json` - Special tokens mapping file

## Usage Instructions

After downloading the model, use the following path in your code to load the model:

```python
model_path = "trained_esm_model"
model = EsmForSequenceClassification.from_pretrained(model_path)
tokenizer = EsmTokenizer.from_pretrained(model_path)
```
