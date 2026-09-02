# kmer-ord
## A tool for pre-assembly deconvolution of complex genomic mixtures using dimensionality reduction of k-mer profiles 

This repository provides a reference-free workflow for deconvoluting long-read sequencing datasets prior to genome assembly. The approach uses k-mer frequency profiles combined with dimensionality-reduction (DR) methods to partition sequencing reads into genome-specific bins without relying on reference databases or trained models.

`kmer-ord` employs modern DR techniques, including t-SNE, UMAP, TriMAP, PacMAP, and LocalMAP, to allow to effectively separate reads originating from multiple eukaryotic nuclear genomes, organelles, symbionts, and associated microbial communities. Local-structure-preserving methods (e.g. UMAP, t-SNE) often resolve reads along continuous trajectories corresponding to chromosomal structure, while global-structure-preserving methods (e.g. TriMAP) are well suited for distinguishing species-level differences in complex samples.

The resulting bins can be assembled independently (“bin-then-assemble”), enabling targeted genome reconstruction and improved assembly quality from mixed or symbiotic samples where physical separation is impractical.

Documentation and tutorials for kmer-ord https://fdboever.github.io/kmer-ord-docs/

## Overview
![Workflow overview](images/overview.png)

## Install

### 1. Clone the repository 

```bash
git clone <repo-url>
cd kmer-ord
```

### 2. Create a fresh conda environment

Option A (recommended): use the provided `environment.yml`

```bash
conda env create -f environment.yml
conda activate kmerord-env
```

Option B: manual setup

```bash
conda create -n kmerord-env python=3.11 -c conda-forge
conda activate kmerord-env

conda install -c conda-forge numpy pandas scikit-learn umap-learn pacmap numba llvmlite biopython typer libspatialite python-igraph hnswlib hdbscan scipy leidenalg matplotlib seaborn setuptools==65.5.0
```

Tip: You can replace `conda` with `mamba` for faster installs.

### 3. Install the package
First ensure you are inside the kmer-ord directory, and activated the conda environment

For users

```bash
pip install .
```

For developers (editable install)

```bash
pip install -e .
```

### 4. Set up external tools and databases

Finally, use kmer-ord to set up internal environments for external tools and downloading rRNA databases (this can take a while, so consider grabbing yourself a coffee) 

```bash
kmer-ord setup
```

### 5. Verify installation 

```bash
kmer-ord --help
```

## Usage

Each `project` / `cluster` / `dr` run writes wall time and peak RAM per stage (and per inner step) to `{output}/benchmarking/benchmark_log.tsv`.

Large-matrix recipe — PCA-pre with IncrementalPCA before DR:

```bash
kmer-ord project -i reads.fastq -o out --pca-pre --keep-pcs 50 --pca-pre-method ipca
```