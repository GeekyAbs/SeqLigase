# SeqLigase BiLSTM — exported model

- Source checkpoint: `/kaggle/working/checkpoints/seqligase_bilstm_best.pt`
- Exported (UTC): 2026-11-05T13:09:56.417586+00:00
- Vocabulary size: 35
- Model config: `{"vocab_size": 35, "embed_dim": 192, "hidden_dim": 384, "num_layers": 3, "dropout": 0.2, "pad_idx": 0}`

## Test metrics

```json
{
  "loss": 0.5742715240616763,
  "token_accuracy": 0.7647849462365591,
  "top3_accuracy": 0.9691358024691358,
  "token_precision": 0.5363677418349599,
  "char_accuracy": 0.9727777366347028,
  "structural_pass_rate": 0.7278161778441995,
  "validity_rate": 0.9295722025032692,
  "n_tokens": 20088.0,
  "n_sequences": 10706.0
}
```

## Loading

```python
from models.bilstm import SeqLigaseBiLSTM
model, bundle = SeqLigaseBiLSTM.from_checkpoint('artifacts/exports/seqligase_bilstm_export.pt')
model.eval()
```
