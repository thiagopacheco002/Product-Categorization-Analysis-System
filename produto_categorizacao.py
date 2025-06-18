import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configurações
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend('agg')

# Função para ler dados
def ler_arquivo_csv(nome_arquivo: str, sep: Optional[str] = None) -> pd.DataFrame:
    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
    separadores = [sep] if sep else [';', ',']
    for encoding in encodings:
        for separator in separadores:
            try:
                return pd.read_csv(nome_arquivo, sep=separator, encoding=encoding, low_memory=False)
            except Exception:
                continue
    raise Exception(f"Não foi possível ler o arquivo {nome_arquivo}")

# Carregando os dados
df_produtos_ativos = ler_arquivo_csv('Base produtos ativos.csv')
df_dados_produtos = ler_arquivo_csv('Base dados produtos.csv')
df_ncm = ler_arquivo_csv('Tabela_NCM_Vigente_Tratado.csv')

df_produtos_ativos.columns = [col.strip() for col in df_produtos_ativos.columns]
df_dados_produtos.columns = [col.strip() for col in df_dados_produtos.columns]

print("Análise Inicial dos Dados:")
print(f"Produtos Ativos: {df_produtos_ativos.shape[0]} registros")
print(f"Base de Dados Produtos: {df_dados_produtos.shape[0]} registros")
print(f"Tabela NCM: {df_ncm.shape[0]} registros")

# Normalizando códigos para fazer merge
df_produtos_ativos['CODIGO'] = df_produtos_ativos['CODIGO'].astype(str).str.strip()
df_dados_produtos['Codigo'] = df_dados_produtos['Codigo'].astype(str).str.strip()

# Merge otimizado para trazer NCM
df_analise = df_produtos_ativos.merge(
    df_dados_produtos[['Codigo', 'NCM', 'Grupo Trib', 'Gr. Compras', 'Selo Inmetro', 'CEST', 'Departamento', 'Grupo', 'Montagem']],
    left_on='CODIGO', right_on='Codigo', how='left'
)
df_analise['NCM'] = df_analise['NCM'].fillna('')
df_analise['Grupo Trib'] = df_analise['Grupo Trib'].fillna('')
df_analise['Gr. Compras'] = df_analise['Gr. Compras'].fillna('')
df_analise['Selo Inmetro'] = df_analise['Selo Inmetro'].fillna('')
df_analise['CEST'] = df_analise['CEST'].fillna('')
df_analise['Departamento'] = df_analise['Departamento'].fillna('')
df_analise['Grupo'] = df_analise['Grupo'].fillna('')
df_analise['Montagem'] = df_analise['Montagem'].fillna('')

# Limpeza de descrição
def extrair_termos_chave(descricao: Optional[str]) -> str:
    if isinstance(descricao, str):
        descricao = descricao.lower()
        descricao = re.sub(r'[^\w\s]', ' ', descricao)
        return descricao
    return ""

df_analise['DESC_LIMPA'] = df_analise['DESCRICAO'].apply(extrair_termos_chave)

# Embeddings com Sentence-BERT
from sentence_transformers import SentenceTransformer

print("Carregando modelo Sentence-BERT...")
modelo_embedding = SentenceTransformer('paraphrase-MiniLM-L6-v2')

print("Gerando embeddings das descrições...")
desc_list = df_analise['DESC_LIMPA'].fillna("").tolist()
embeddings = modelo_embedding.encode(desc_list, show_progress_bar=True)
df_analise['EMBEDDING'] = list(embeddings)

# Embedding médio por família
print("Calculando embedding médio por família...")
familia2embedding = {}
for familia in df_analise['FAMILIA'].unique():
    idxs = df_analise['FAMILIA'] == familia
    if idxs.sum() > 0:
        familia_embeds = np.stack(df_analise.loc[idxs, 'EMBEDDING'])
        familia2embedding[familia] = familia_embeds.mean(axis=0)
    else:
        familia2embedding[familia] = np.zeros(embeddings[0].shape)

def similaridade_embedding(row):
    emb_produto = row['EMBEDDING']
    emb_familia = familia2embedding.get(row['FAMILIA'], np.zeros_like(emb_produto))
    return float(cosine_similarity([emb_produto], [emb_familia])[0][0])

df_analise['SIMILARIDADE_EMBEDDING'] = df_analise.apply(similaridade_embedding, axis=1)

# Use a similaridade de embedding como score principal
df_analise['SIMILARIDADE'] = df_analise['SIMILARIDADE_EMBEDDING']

# Função para encontrar as palavras mais comuns por família
def palavras_comuns_por_familia(df: pd.DataFrame, familia_codigo: str) -> Dict[str, int]:
    subset = df[df['FAMILIA'] == familia_codigo]
    if subset.empty:
        return {}
    todas_desc = " ".join(subset['DESC_LIMPA'].fillna(""))
    palavras = todas_desc.split()
    todas_familias = df['FAMILIA'].unique()
    palavras_familias = {palavra: df[df['DESC_LIMPA'].str.contains(palavra)]['FAMILIA'].nunique() for palavra in set(palavras)}
    contador = Counter(palavras)
    palavras_filtradas = {k: v for k, v in contador.items() if len(k) > 2 and v > 1 and palavras_familias[k] < len(todas_familias) * 0.3}
    return palavras_filtradas

# Criando perfis de família
def criar_perfil_familia(df: pd.DataFrame) -> Dict[str, Dict[str, Union[str, Dict[str, int]]]]:
    familias = df['FAMILIA'].unique()
    perfis = {}
    for familia in familias:
        subset = df[df['FAMILIA'] == familia]
        desc_familia = subset['DESC. FAMILIA'].iloc[0] if not subset.empty else ""
        palavras_comuns = palavras_comuns_por_familia(df, familia)
        ncm_comum = subset['NCM'].mode()[0] if not subset['NCM'].mode().empty else ""
        grupo_trib = subset['Grupo Trib'].mode()[0] if not subset['Grupo Trib'].mode().empty else ""
        gr_compras = subset['Gr. Compras'].mode()[0] if not subset['Gr. Compras'].mode().empty else ""
        selo_inmetro = subset['Selo Inmetro'].mode()[0] if not subset['Selo Inmetro'].mode().empty else ""
        cest = subset['CEST'].mode()[0] if not subset['CEST'].mode().empty else ""
        departamento = subset['Departamento'].mode()[0] if not subset['Departamento'].mode().empty else ""
        grupo = subset['Grupo'].mode()[0] if not subset['Grupo'].mode().empty else ""
        montagem = subset['Montagem'].mode()[0] if not subset['Montagem'].mode().empty else ""
        perfis[familia] = {
            'desc_familia': desc_familia,
            'palavras_comuns': palavras_comuns,
            'ncm_comum': ncm_comum,
            'grupo_trib': grupo_trib,
            'gr_compras': gr_compras,
            'selo_inmetro': selo_inmetro,
            'cest': cest,
            'departamento': departamento,
            'grupo': grupo,
            'montagem': montagem
        }
    return perfis

print("Criando perfis de família...")
perfis_familias = criar_perfil_familia(df_analise)

# Pré-calcula os valores de perfil para cada família (para vetorização)
familia2ncm = {fam: perfil['ncm_comum'] for fam, perfil in perfis_familias.items()}
familia2grupo_trib = {fam: perfil['grupo_trib'] for fam, perfil in perfis_familias.items()}
familia2gr_compras = {fam: perfil['gr_compras'] for fam, perfil in perfis_familias.items()}
familia2selo_inmetro = {fam: perfil['selo_inmetro'] for fam, perfil in perfis_familias.items()}
familia2cest = {fam: perfil['cest'] for fam, perfil in perfis_familias.items()}
familia2departamento = {fam: perfil['departamento'] for fam, perfil in perfis_familias.items()}
familia2grupo = {fam: perfil['grupo'] for fam, perfil in perfis_familias.items()}
familia2montagem = {fam: perfil['montagem'] for fam, perfil in perfis_familias.items()}
familia2palavras = {fam: set(perfil['palavras_comuns'].keys()) for fam, perfil in perfis_familias.items()}

# Vetorizando critérios simples
df_analise['NCM_MATCH'] = df_analise.apply(lambda row: int(row['NCM'] == familia2ncm.get(row['FAMILIA'], "")), axis=1)
df_analise['GRUPO_TRIB_MATCH'] = df_analise.apply(lambda row: int(row['Grupo Trib'] == familia2grupo_trib.get(row['FAMILIA'], "")), axis=1)
df_analise['GR_COMPRAS_MATCH'] = df_analise.apply(lambda row: int(row['Gr. Compras'] == familia2gr_compras.get(row['FAMILIA'], "")), axis=1)
df_analise['SELO_INMETRO_MATCH'] = df_analise.apply(lambda row: int(row['Selo Inmetro'] == familia2selo_inmetro.get(row['FAMILIA'], "")), axis=1)
df_analise['CEST_MATCH'] = df_analise.apply(lambda row: int(row['CEST'] == familia2cest.get(row['FAMILIA'], "")), axis=1)
df_analise['DEPARTAMENTO_MATCH'] = df_analise.apply(lambda row: int(row['Departamento'] == familia2departamento.get(row['FAMILIA'], "")), axis=1)
df_analise['GRUPO_MATCH'] = df_analise.apply(lambda row: int(row['Grupo'] == familia2grupo.get(row['FAMILIA'], "")), axis=1)
df_analise['MONTAGEM_MATCH'] = df_analise.apply(lambda row: int(row['Montagem'] == familia2montagem.get(row['FAMILIA'], "")), axis=1)

# Palavras-chave: otimizado
def score_palavras(desc, familia):
    if not isinstance(desc, str): return 0.0
    palavras = familia2palavras.get(familia, set())
    return min(sum(1 for p in palavras if p in desc), 2.0)  # Peso máximo 2.0

df_analise['PALAVRAS_SCORE'] = df_analise.apply(lambda row: score_palavras(row['DESC_LIMPA'], row['FAMILIA']), axis=1)

# Score final vetorizado
peso_palavras = 2.25
peso_ncm = 1.25  # Ajustado de 2.0 para 1.0
peso_grupo_trib = 0.5
peso_gr_compras = 0.5
peso_selo_inmetro = 0.5
peso_cest = 0.5
peso_departamento = 0.5
peso_grupo = 0.5 
peso_montagem = 0.5
max_score = (
    peso_palavras + peso_ncm + peso_grupo_trib + peso_gr_compras +
    peso_selo_inmetro + peso_cest + peso_departamento + peso_grupo + peso_montagem
)

df_analise['SIMILARIDADE'] = (
    df_analise['PALAVRAS_SCORE'] * (peso_palavras / 2.0) +
    df_analise['NCM_MATCH'] * peso_ncm +
    df_analise['GRUPO_TRIB_MATCH'] * peso_grupo_trib +
    df_analise['GR_COMPRAS_MATCH'] * peso_gr_compras +
    df_analise['SELO_INMETRO_MATCH'] * peso_selo_inmetro +
    df_analise['CEST_MATCH'] * peso_cest +
    df_analise['DEPARTAMENTO_MATCH'] * peso_departamento +
    df_analise['GRUPO_MATCH'] * peso_grupo +
    df_analise['MONTAGEM_MATCH'] * peso_montagem
) / max_score

# Vetoriza as descrições
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_analise['DESC_LIMPA'].fillna(""))

# Agora aplique o SVD
vectorizer = TruncatedSVD(n_components=2, random_state=42)
pca_matrix = vectorizer.fit_transform(tfidf_matrix)
df_analise['PCA1'] = pca_matrix[:, 0]
df_analise['PCA2'] = pca_matrix[:, 1]

# Identificando possíveis produtos mal categorizados
threshold = 0.6
produtos_suspeitos = df_analise[df_analise['SIMILARIDADE'] < threshold].copy()

# Encontrando família sugerida para cada produto suspeito (mantém apply, pois é análise cruzada)
def encontrar_familia_sugerida(produto: pd.Series) -> (Optional[str], float):
    emb_produto = produto['EMBEDDING']
    desc_produto = produto['DESC_LIMPA']
    melhor_score = -1.0
    familia_sugerida = None

    # Novo: armazena famílias candidatas por número de palavras-chave coincidentes
    candidatos = []
    for familia, emb_familia in familia2embedding.items():
        palavras_familia = familia2palavras.get(familia, set())
        n_palavras = sum(1 for palavra in palavras_familia if palavra in desc_produto)
        score = float(cosine_similarity([emb_produto], [emb_familia])[0][0])
        candidatos.append((familia, n_palavras, score))

    # Prioriza famílias com mais palavras-chave coincidentes, depois maior score
    candidatos.sort(key=lambda x: (x[1], x[2]), reverse=True)
    familia_sugerida, _, melhor_score = candidatos[0]
    return familia_sugerida, melhor_score

import re
# ...outros imports...

def numeros_na_string(s):
    """Extrai todos os números de uma string como um set de strings."""
    return set(re.findall(r'\d+', s or ""))

# Progresso com tqdm
resultados = []
if not produtos_suspeitos.empty:
    print(f"Iniciando análise de {len(produtos_suspeitos)} produtos suspeitos...")
    for idx, (_, produto) in enumerate(tqdm(produtos_suspeitos.iterrows(), total=len(produtos_suspeitos), desc="Processando")):
        familia_atual = produto['FAMILIA']
        familia_sugerida, score_sugerido = encontrar_familia_sugerida(produto)
        desc_atual = perfis_familias.get(familia_atual, {}).get('desc_familia', "")
        desc_sugerida = perfis_familias.get(familia_sugerida, {}).get('desc_familia', "")
        num_atual = numeros_na_string(desc_atual)
        num_sugerida = numeros_na_string(desc_sugerida)
        num_produto = numeros_na_string(produto['DESCRICAO'])

        # Só sugere se o número da família sugerida aparecer na descrição do produto
        if (
            familia_sugerida and 
            familia_sugerida != familia_atual and 
            (score_sugerido - produto['SIMILARIDADE']) > 0.10 and
            (not num_sugerida or num_sugerida & num_produto)
        ):
            motivo = (
                f"Produto possui baixa similaridade ({produto['SIMILARIDADE']:.2f}) com a família atual "
                f"e apresenta maior similaridade ({score_sugerido:.2f}) com a família sugerida '{familia_sugerida}' (baseado em embeddings)."
            )
            resultados.append({
                'CODIGO': produto['CODIGO'],
                'DESCRICAO': produto['DESCRICAO'],
                'FAMILIA_ATUAL': familia_atual,
                'DESC_FAMILIA_ATUAL': desc_atual,
                'FAMILIA_SUGERIDA': familia_sugerida,
                'DESC_FAMILIA_SUGERIDA': desc_sugerida,
                'SCORE_ATUAL': produto['SIMILARIDADE'],
                'SCORE_SUGERIDO': score_sugerido,
                'MOTIVO': motivo
            })

df_resultados = pd.DataFrame(resultados)
if not df_resultados.empty:
    df_resultados.to_csv('produtos_com_possivel_erro_categorizacao.csv', index=False, sep=';', encoding='utf-8-sig')
    print(f"\nForam encontrados {len(df_resultados)} produtos com possível erro de categorização.")
    print("Os resultados foram salvos no arquivo 'produtos_com_possivel_erro_categorizacao.csv'")
else:
    print("\nNenhum produto com possível erro de categorização foi encontrado com o threshold atual.")

# Visualização
def visualizar_distribuicao_similaridade(df_analise: pd.DataFrame, threshold: float) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_analise, x='SIMILARIDADE', bins=20, kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title('Distribuição de Similaridade dos Produtos com suas Famílias')
    plt.xlabel('Score de Similaridade')
    plt.ylabel('Frequência')
    plt.legend()
    plt.savefig('distribuicao_similaridade.png')
    print("Gráfico de distribuição de similaridade salvo como 'distribuicao_similaridade.png'")

def visualizar_agrupamentos(df_analise: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_analise,
        x='PCA1', 
        y='PCA2', 
        hue='FAMILIA', 
        size='SIMILARIDADE',
        sizes=(20, 200),
        palette='viridis'
    )
    plt.title('Visualização de Produtos por Família (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.savefig('agrupamentos_pca.png')
    print("Gráfico de agrupamentos PCA salvo como 'agrupamentos_pca.png'")

try:
    visualizar_distribuicao_similaridade(df_analise, threshold)
    visualizar_agrupamentos(df_analise)
except Exception as e:
    print(f"Erro ao gerar visualizações: {e}")

def gerar_estatisticas_familia(df_analise: pd.DataFrame) -> None:
    stats = df_analise.groupby('FAMILIA').agg({
        'DESC. FAMILIA': 'first',
        'SIMILARIDADE': ['mean', 'min', 'count']
    })
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={
        'DESC. FAMILIA_first': 'DESCRICAO_FAMILIA',
        'SIMILARIDADE_mean': 'SIMILARIDADE_MEDIA',
        'SIMILARIDADE_min': 'SIMILARIDADE_MINIMA',
        'SIMILARIDADE_count': 'QTD_PRODUTOS'
    })
    stats = stats.sort_values('SIMILARIDADE_MEDIA')
    stats.to_csv('estatisticas_por_familia.csv', index=True, sep=';', encoding='utf-8-sig')
    print("Estatísticas por família salvas no arquivo 'estatisticas_por_familia.csv'")

gerar_estatisticas_familia(df_analise)

# --- Clustering interno por família usando DBSCAN nos embeddings ---

def aplicar_dbscan_por_familia(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 3) -> pd.DataFrame:
    """
    Aplica DBSCAN nos embeddings de cada família e retorna o DataFrame com coluna 'CLUSTER_FAMILIA'
    - eps: raio de vizinhança (ajuste conforme necessário)
    - min_samples: mínimo de pontos para formar um cluster
    """
    df = df.copy()
    cluster_labels = np.full(len(df), -99)  # valor default para produtos sem família definida
    for familia in df['FAMILIA'].unique():
        idxs = df['FAMILIA'] == familia
        if idxs.sum() < min_samples:
            continue  # não clusteriza famílias muito pequenas
        X = np.stack(df.loc[idxs, 'EMBEDDING'].values)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X)
        cluster_labels[idxs] = labels
    df['CLUSTER_FAMILIA'] = cluster_labels
    return df

print("Aplicando DBSCAN por família para análise de coesão interna...")
df_analise = aplicar_dbscan_por_familia(df_analise, eps=0.35, min_samples=3)

# Estatísticas de clusterização
def estatisticas_clusters(df: pd.DataFrame):
    stats = []
    for familia in df['FAMILIA'].unique():
        sub = df[df['FAMILIA'] == familia]
        n_prod = len(sub)
        n_noise = (sub['CLUSTER_FAMILIA'] == -1).sum()
        n_clusters = len(set(sub['CLUSTER_FAMILIA'])) - (1 if -1 in sub['CLUSTER_FAMILIA'].values else 0)
        stats.append({
            'FAMILIA': familia,
            'QTD_PRODUTOS': n_prod,
            'QTD_CLUSTERS': n_clusters,
            'QTD_NOISE': n_noise,
            'PCT_NOISE': n_noise / n_prod if n_prod > 0 else 0
        })
    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.sort_values('PCT_NOISE', ascending=False)
    df_stats.to_csv('estatisticas_clusters_familia.csv', index=False, sep=';', encoding='utf-8-sig')
    print("Estatísticas de clusters por família salvas em 'estatisticas_clusters_familia.csv'")

estatisticas_clusters(df_analise)

print("\nAnálise de categorização de produtos concluída!")

# Carrega estatísticas de clusters
df_clusters = pd.read_csv('estatisticas_clusters_familia.csv', sep=';')

# Define famílias críticas (exemplo: mais de 20% de noise)
familias_criticas = df_clusters[df_clusters['PCT_NOISE'] > 0.2]['FAMILIA'].tolist()

# Marque produtos dessas famílias para revisão prioritária
df_analise['FAMILIA_CRITICA'] = df_analise['FAMILIA'].isin(familias_criticas)

# Marque produtos outliers
df_analise['OUTLIER_FAMILIA'] = df_analise['CLUSTER_FAMILIA'] == -1

# Score ajustado: penaliza outlier e famílias críticas
df_analise['SIMILARIDADE_AJUSTADA'] = df_analise['SIMILARIDADE']
df_analise.loc[df_analise['OUTLIER_FAMILIA'], 'SIMILARIDADE_AJUSTADA'] *= 0.7  # penaliza outlier
df_analise.loc[df_analise['FAMILIA_CRITICA'], 'SIMILARIDADE_AJUSTADA'] *= 0.85  # penaliza família crítica

produtos_prioritarios = df_analise[
    (df_analise['SIMILARIDADE_AJUSTADA'] < 0.6) &
    (df_analise['OUTLIER_FAMILIA'] | df_analise['FAMILIA_CRITICA'])
]
produtos_prioritarios.to_csv('produtos_prioritarios_revisao.csv', index=False, sep=';', encoding='utf-8-sig')
print(f"Foram encontrados {len(produtos_prioritarios)} produtos prioritários para revisão (outlier/família crítica e baixa similaridade).")

def numeros_na_string(s):
    """Extrai todos os números de uma string como um set de strings."""
    return set(re.findall(r'\d+', s or ""))