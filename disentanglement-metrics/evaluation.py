from src.metrics.dci import dci  # 假设代码保存为 dci.py
import scanpy as sc
import pandas as pd
import gc
# 初始化结果列表
all_results = []
# , '9_Mouse_Thymus','10_Mouse_Breast_Tumor'
# for dataset in ['4_Human_Lymph_Node', '9_Mouse_Thymus','10_Mouse_Breast_Tumor']:
    # adata1 = sc.read_h5ad('../Dataset/Dimension reduction results/Dimension reduction results/'+ dataset +'/adata_RNA_processed.h5ad')
    # adata2 = sc.read_h5ad('../Dataset/Dimension reduction results/Dimension reduction results/'+ dataset +'/adata_ADT_processed.h5ad')
adata1 = sc.read_h5ad('../Dataset/result/adata_omics1.h5ad')
adata2 = sc.read_h5ad('../Dataset/result/adata_omics2.h5ad')
# 读取 CSV 文件
# file_path = '../Dataset/Dimension reduction results/Dimension reduction results/'+ dataset +'/SpaMV_z_merged.csv'
file_path = '../Dataset/result/spot_topic.csv'
df = pd.read_csv(file_path, index_col=0)

# 提取 Transcriptomics private topics 列
# transcriptomics_private = df.filter(like="Transcriptomics private", axis=1).values

# 提取 Proteomics private topics 列
# proteomics_private = df.filter(like="Proteomics private", axis=1).values
# share = df.filter(like="Shared topic", axis=1).values
omics1_private = df.filter(like="Omics 1 private", axis=1).values
omics2_private = df.filter(like="Omics 2 private", axis=1).values
share = df.filter(like="Shared", axis=1).values


adata1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X
adata2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else adata2.X

# , share
# for factors_type in [transcriptomics_private, proteomics_private]:
for factors_type in [share]:
    # adata1, adata2,
    # for codes_type in [share]:
    # for codes_type in [transcriptomics_private, proteomics_private]:
    for codes_type in [omics1_private, omics2_private]:
        # Convert Series to 2D arrays
        factors = factors_type
        codes = codes_type

        # Call the dci function
        disentanglement_A_in_B, completeness_A_in_B, informativeness_A_in_B = dci(
            factors=factors,
            codes=codes,
            model="random_forest"
        )

        print(# f"Dataset: {dataset}, "
              f"Factors type: {factors_type.shape}, "
              f"Codes type: {codes_type.shape}, "                                       
              f"Modularity: {disentanglement_A_in_B:.4f}, "
              f"Compactness: {completeness_A_in_B:.4f}, "                                                
              f"Informativeness: {informativeness_A_in_B:.4f}")
        all_results.append({
            # "Dataset": dataset,
            "Factors type": factors_type.shape,
            "Codes type": codes_type.shape,
            "Modularity": disentanglement_A_in_B,
            "Compactness": completeness_A_in_B,
            "Informativeness": informativeness_A_in_B,
        })
# del adata1, adata2, df, transcriptomics_private, proteomics_private, share
del adata1, adata2, df, omics1_private, omics2_private, share
gc.collect()

# results_df = pd.DataFrame(all_results)
# results_df.to_csv('results.csv', index=False)
