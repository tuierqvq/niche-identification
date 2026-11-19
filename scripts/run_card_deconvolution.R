suppressPackageStartupMessages({
  library(Matrix)
  library(CARD)
})

args <- commandArgs(trailingOnly = TRUE)
if (identical(args, "--check")) {
  quit(save = "no", status = 0)
}
if (length(args) != 11) {
  stop("Usage: run_card_deconvolution.R spatial.mtx spatial_genes.txt spots.txt coords.csv reference.mtx reference_genes.txt cells.txt cell_metadata.csv output.csv minCountGene minCountSpot")
}

read_counts <- function(matrix_path, genes_path, observations_path) {
  # Matrix Market stores genes x observations; names must preserve Python export order.
  counts <- as(Matrix::readMM(matrix_path), "CsparseMatrix")
  genes <- readLines(genes_path, warn = FALSE)
  observations <- readLines(observations_path, warn = FALSE)
  if (nrow(counts) != length(genes) || ncol(counts) != length(observations)) {
    stop("Matrix dimensions do not match gene/observation files.")
  }
  if (anyDuplicated(genes) || anyDuplicated(observations)) {
    stop("Gene and observation names must be unique.")
  }
  rownames(counts) <- genes
  colnames(counts) <- observations
  counts
}

spatial_count <- read_counts(args[[1]], args[[2]], args[[3]])
spatial_location <- read.csv(args[[4]], row.names = 1, check.names = FALSE)
if (!all(c("x", "y") %in% colnames(spatial_location)) ||
    !setequal(rownames(spatial_location), colnames(spatial_count))) {
  stop("Spatial coordinates must contain x/y and exactly match the spot IDs.")
}
spatial_location <- spatial_location[colnames(spatial_count), c("x", "y"), drop = FALSE]

sc_count <- read_counts(args[[5]], args[[6]], args[[7]])
sc_meta <- read.csv(args[[8]], stringsAsFactors = FALSE, check.names = FALSE,
                    colClasses = "character", na.strings = character())
if (!all(c("cell_id", "cell_type", "sample_id") %in% colnames(sc_meta)) ||
    anyDuplicated(sc_meta$cell_id) || !setequal(sc_meta$cell_id, colnames(sc_count))) {
  stop("Reference metadata must contain cell_id/cell_type/sample_id and match reference cells.")
}
rownames(sc_meta) <- sc_meta$cell_id
sc_meta <- sc_meta[colnames(sc_count), c("cell_type", "sample_id"), drop = FALSE]

card_obj <- createCARDObject(
  sc_count = sc_count,
  sc_meta = sc_meta,
  spatial_count = spatial_count,
  spatial_location = spatial_location,
  ct.varname = "cell_type",
  ct.select = unique(sc_meta$cell_type),
  sample.varname = "sample_id",
  minCountGene = as.numeric(args[[10]]),
  minCountSpot = as.numeric(args[[11]])
)
card_obj <- CARD_deconvolution(CARD_object = card_obj)

prop <- as.data.frame(card_obj@Proportion_CARD, check.names = FALSE)
prop <- cbind(spot_id = rownames(prop), prop)
write.csv(prop, args[[9]], row.names = FALSE, quote = TRUE)
