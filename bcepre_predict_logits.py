import pandas as pd
from transformers import EsmForSequenceClassification, EsmTokenizer
import torch
from tqdm import tqdm
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "trained_esm_model"
model = EsmForSequenceClassification.from_pretrained(model_path)
model.to(device) 

tokenizer = EsmTokenizer.from_pretrained(model_path)

def read_fasta(fasta_file):
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])
    return sequence

def sliding_window_prediction(sequence, min_window_size=5, batch_size=8):
    predictions = []
    max_window_size = len(sequence)
    
    for window_size in range(min_window_size, max_window_size + 1):
        windows = []
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            windows.append(window)
        
        for start_idx in tqdm(range(0, len(windows), batch_size)):
            batch_windows = windows[start_idx:start_idx + batch_size]
            inputs = tokenizer(batch_windows, return_tensors='pt', padding=True, truncation=True, max_length=window_size)
            
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()  
                batch_logits = logits.cpu().numpy()  

            for idx, window in enumerate(batch_windows):
                predictions.append({
                    "sequence": window,
                    "window_size": window_size,
                    "prediction": batch_predictions[idx],
                    "logits": batch_logits[idx].tolist()  
                })
    
    return predictions

def save_predictions_to_csv(predictions, output_file):
    df = pd.DataFrame(predictions)
    if df['logits'].apply(lambda x: len(x)).nunique() > 1:
        pass
    else:
        logits_df = pd.DataFrame(df['logits'].tolist())
        logits_df.columns = [f'logit_{i}' for i in range(logits_df.shape[1])]
        df = pd.concat([df.drop('logits', axis=1), logits_df], axis=1)
    
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    fasta_file = 'example_data/test_aa.fa'  # Input FASTA file path
    sequence = read_fasta(fasta_file)

    predictions = sliding_window_prediction(sequence, min_window_size=5, batch_size=8)

    output_file = 'predictions/PDCoV_GDSG10_RBD_aa_logits.csv'  # Output CSV file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure output directory exists
    save_predictions_to_csv(predictions, output_file)

if __name__ == "__main__":
    main()
