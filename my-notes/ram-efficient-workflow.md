RAM-Efficient Workflow

Input
↓
Convert to Rₘ
↓
Rₘ data
↓
K-means counting ✓ RAM efficient
↓
Raw k-means stats ← Needs fixing: Duplicated matrix allocations currently
↓
Matrix preprocessing ← Issue: k-means matrix loaded in RAM twice currently
↓
Normalization ← Fix to use standardized-log data + CLR identity to avoid unnecessary temp arrays
↓
Pre-PCA ← Use incremental PCA to avoid PCA internal float32 copy. Reduce to ≥100 PCA, I think
↓
Hyperparameter + DR methods

Additional notes
For small accumulation error that adds:
n_reads × n_samples × n_features x seq_id_string_bytes

With the changes above, the program should be able to get:
~ n_reads x 4^k × 4 bytes
For the peak RAM cost of the k-means matrix (pre-canonical-correction)
After Pre-PCA, the downstream DR methods should be memory safe on the large datasets.
