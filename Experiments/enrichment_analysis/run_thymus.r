library(GSVA)
library(GSEABase)
library(limma)
library(Biobase)
library(ggplot2)
library(ggpubr)
library(ggsci)

m2 <- getGmt("/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/MSigDB/m2.all.v2024.1.Mm.symbols.gmt")
m5 <- getGmt("/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/MSigDB/m5.all.v2024.1.Mm.symbols.gmt")


in_path <- "/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/data_in/yangv2/Mouse Thymus results (RNA+Protein)/mouse_thymus_transcriptomics_match"
outDir <- "/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/data_in/yangv2/Mouse Thymus results (RNA+Protein)/out/protein"
filename <- "mouse_thymus"

# in_path <- "/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/data_in/yangv2/Mouse Thymus results (RNA+Protein)/mouse_thymus_transcriptomics_match"
# outDir <- "/maiziezhou_lab2/yunfei/Projects/Spinal_MERFISH/GSVA/data_in/yangv2/Mouse Thymus results (RNA+Protein)/out/rna"
# filename <- "mouse_thymus"


# Load the CSV files
expression_matrix <- read.csv(file.path(in_path, "expression_matrix.csv"), row.names = 1)
phenotype_data   <- read.csv(file.path(in_path, "phenotype_data.csv"), row.names = 1)
feature_data     <- read.csv(file.path(in_path, "feature_data.csv"), row.names = 1)

# Convert phenotype and feature data into AnnotatedDataFrame
phenoData <- AnnotatedDataFrame(data = phenotype_data)
featureData <- AnnotatedDataFrame(data = feature_data)

# Create the ExpressionSet object
eset <- ExpressionSet(
  assayData = as.matrix(expression_matrix),
  phenoData = phenoData,
  featureData = featureData
)

counts <- exprs(eset)
# sc.eset[] <- lapply(sc.eset, as.integer)

gsva_params <- gsvaParam(
    exprData=as.matrix(counts),
    geneSets=m2,
    kcdf = 'Gaussian',
    minSize = 5,
)
reactome_enrichscore <- gsva(gsva_params)
idx <- order(rowMeans(reactome_enrichscore), decreasing = T)
reactome_enrichscore <- reactome_enrichscore[idx,]

write.table(reactome_enrichscore, file = file.path(outDir, paste0(filename, "_m2.csv")), quote = FALSE, sep = ",")

gsva_params <- gsvaParam(
    exprData=as.matrix(counts),
    geneSets=m5,
    kcdf = 'Gaussian',
    minSize = 5,
)
GO_enrichscore <- gsva(gsva_params)
write.table(GO_enrichscore, file = file.path(outDir, paste0(filename, "_m5.csv")), quote = FALSE, sep = ",")
