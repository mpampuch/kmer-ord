# Minimum tests and with minimum parameters to run:

```bash
# TEST 1: K = 6
kmer-ord project \
 --input TEST-DATA/63_Monoraphidiumcircinale.hifi_reads.subsampled.1percent.fasta \
 --output "TEST/K6/$(date -u +%Y%m%d_%H%M%S)" \
 --threads 1 \
 --kmer 6 \
 --tiara --dr umap,tsne,trimap,pacmap,localmap,pca,sparse_pca,kernel_pca,lle

# TEST 2: K = 8
kmer-ord project \
 --input TEST-DATA/63_Monoraphidiumcircinale.hifi_reads.subsampled.1percent.fasta \
 --output "TEST/K10/$(date -u +%Y%m%d_%H%M%S)" \
 --threads 1 \
 --kmer 8 \
 --tiara --dr umap,tsne,trimap,pacmap,localmap,pca,sparse_pca,kernel_pca,lle
```
