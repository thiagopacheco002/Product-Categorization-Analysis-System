# Product Categorization Analysis System

## Overview

This project implements a system for analyzing and validating product categorization using machine learning and natural language processing techniques. The primary goal is to identify potentially miscategorized products and suggest recategorizations based on similarity analysis.

## Features

### Multi-criteria Similarity Analysis
- Combines semantic embeddings with technical characteristics (NCM, tax group, CEST)
- Family-specific keyword analysis
- Composite scoring considering multiple classification factors

### Semantic Processing
- Uses Sentence-BERT (paraphrase-MiniLM-L6-v2) for embedding generation
- Cosine similarity calculation between products and families
- Semantic cohesion analysis within each category

### Clustering and Anomaly Detection
- DBSCAN clustering for internal cohesion analysis by family
- Identification of outlier products within their categories
- Prioritization of products for manual review

### Reporting and Visualization
- CSV export of analysis results
- Similarity distribution charts
- PCA visualization of product clusters

## Technical Stack

- Python with pandas, numpy, and scikit-learn
- Sentence-Transformers for semantic embeddings
- DBSCAN for unsupervised clustering
- TF-IDF and PCA for dimensional analysis
- Matplotlib and seaborn for visualizations

## Installation

```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn tqdm
```

## Usage

1. Prepare your input CSV files:
   - `Base produtos ativos.csv` - Active products with codes and families
   - `Base dados produtos.csv` - Technical data including NCM and tax info
   - `Tabela_NCM_Vigente_Tratado.csv` - Current NCM reference table

2. Run the analysis:
```bash
python produto_categorizacao.py
```

## Methodology

The system processes product data through multiple stages:

1. **Preprocessing**: Description cleaning and normalization
2. **Embedding Generation**: Vector representation creation using transformers
3. **Profile Creation**: Statistical profile construction per family
4. **Similarity Scoring**: Weighted criteria-based score calculation
5. **Clustering Analysis**: Internal cohesion analysis using DBSCAN
6. **Anomaly Detection**: Low-adherence product identification
7. **Recommendation Engine**: Recategorization suggestions

## Data Input

The system expects three CSV files:
- Active products base with codes and current families
- Technical data base including NCM, taxation, and characteristics
- Current NCM table for reference

## Output

Generates multiple reports for analysis:
- `produtos_com_possivel_erro_categorizacao.csv` - Products with possible categorization errors
- `produtos_prioritarios_revisao.csv` - Priority products for review (outliers and critical families)
- `estatisticas_por_familia.csv` - Quality statistics per family
- `estatisticas_clusters_familia.csv` - Internal clustering metrics
- `distribuicao_similaridade.png` - Similarity distribution chart
- `agrupamentos_pca.png` - PCA visualization of product clusters

## Configuration

Key parameters can be adjusted in the code:

```python
# Similarity threshold for anomaly detection
threshold = 0.6

# Criteria weights
peso_palavras = 2.25
peso_ncm = 1.25
peso_grupo_trib = 0.5
# ... other weights

# DBSCAN parameters
eps = 0.35
min_samples = 3
```

## Applications

This system is useful for organizations that need to:
- Audit and improve product categorization quality
- Verify compliance with fiscal and tax classifications
- Optimize e-commerce catalogs
- Automate product data review processes

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- sentence-transformers
- matplotlib
- seaborn
- tqdm

## Limitations

The system requires structured data and quality textual descriptions. Effectiveness depends on input data consistency and may require parameter adjustments for different product domains.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
