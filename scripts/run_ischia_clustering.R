# ISCHIA composition-clustering bridge. The upstream R module is sourced unchanged.
load_ischia_module <- function(path) {
  required <- c("Matrix", "SeuratObject", "jsonlite")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Missing R packages: ", paste(missing, collapse = ", "))
  if (!file.exists(path)) stop("ISCHIA upstream module is missing: ", path)
  module <- new.env(parent = baseenv())
  module$kmeans <- stats::kmeans
  module$AddMetaData <- SeuratObject::AddMetaData
  module$plot <- graphics::plot
  sys.source(path, envir = module)
  expected <- c("spatial.object", "Celltype_deconvolved_probs", "k")
  if (!is.function(module$Composition.cluster) ||
      !identical(names(formals(module$Composition.cluster)), expected)) {
    stop("Unexpected ISCHIA Composition.cluster interface.")
  }
  module
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2 && args[[1]] == "--check") {
  load_ischia_module(args[[2]])
  cat("ISCHIA Composition.cluster environment OK\n")
  quit(save = "no", status = 0)
}
if (length(args) != 9) {
  stop("Usage: run_ischia_clustering.R composition.csv labels.csv centers.csv metrics.json fit.rds upstream.R k seed elbow_max_k")
}
module <- load_ischia_module(args[[6]])
input <- read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
if (ncol(input) < 3 || names(input)[[1]] != "ischia_spot_id" || anyDuplicated(names(input))) {
  stop("Expected unique columns: ischia_spot_id followed by at least two cell types.")
}
ids <- as.character(input[[1]])
if (anyNA(ids) || any(ids == "") || anyDuplicated(ids)) stop("Spot IDs must be unique and nonempty.")
if (!all(vapply(input[-1], is.numeric, logical(1)))) stop("Cell-type composition must be numeric.")
composition <- as.matrix(input[-1])
rownames(composition) <- ids
if (any(!is.finite(composition)) || any(composition < 0) ||
    any(abs(rowSums(composition) - 1) > 1e-8)) {
  stop("ISCHIA requires finite, nonnegative cell-type proportions with row sums of one.")
}
k <- as.integer(args[[7]])
seed <- as.integer(args[[8]])
elbow_max_k <- as.integer(args[[9]])
if (is.na(k) || k < 1 || k >= nrow(composition) || k > nrow(unique(composition))) {
  stop("Requested k is incompatible with the number of spots or distinct compositions.")
}

# The Seurat object carries metadata; the official function clusters the original matrix.
# This assay contains cell-type proportions, not RNA counts.
object <- SeuratObject::CreateSeuratObject(
  counts = Matrix::Matrix(t(composition), sparse = TRUE), assay = "CellComposition")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
if (elbow_max_k > 0) {
  if (elbow_max_k >= nrow(composition) || elbow_max_k > nrow(unique(composition))) {
    stop("elbow_max_k exceeds the available spots or distinct profiles.")
  }
  set.seed(seed)
  grDevices::pdf(file.path(dirname(args[[2]]), "elbow.pdf"))
  tryCatch(module$Composition.cluster.k(composition, elbow_max_k),
           finally = grDevices::dev.off())
}
# Restore the seed after diagnostics so enabling an elbow plot cannot change fitted labels.
set.seed(seed)
object <- module$Composition.cluster(object, composition, k)
labels <- as.character(object[[]][ids, "CompositionCluster_CC"])
if (anyNA(labels) || length(unique(labels)) != k) stop("ISCHIA returned incomplete cluster labels.")
groups <- split(seq_len(nrow(composition)), labels)
centers <- do.call(rbind, lapply(groups, function(indices) colMeans(composition[indices, , drop = FALSE])))
withinss <- sum((composition - centers[labels, , drop = FALSE]) ^ 2)
metrics <- list(
  method = "ISCHIA Composition.cluster", implementation = "unchanged upstream R module",
  n_spots = nrow(composition), n_cell_types = ncol(composition), n_clusters = k, seed = seed,
  kmeans_algorithm = "Hartigan-Wong", kmeans_nstart = 1L, kmeans_iter_max = 10L,
  total_withinss = withinss, cluster_sizes = as.list(table(labels)),
  r_version = R.version.string, SeuratObject_version = as.character(utils::packageVersion("SeuratObject")))
write.csv(data.frame(ischia_spot_id = ids, CompositionCluster_CC = labels),
          args[[2]], row.names = FALSE, quote = TRUE)
write.csv(data.frame(CompositionCluster_CC = rownames(centers), centers, check.names = FALSE),
          args[[3]], row.names = FALSE, quote = TRUE)
jsonlite::write_json(metrics, args[[4]], pretty = TRUE, auto_unbox = TRUE)
saveRDS(list(spatial_object = object, centers = centers, metrics = metrics), args[[5]])
cat("ISCHIA clustering completed:", nrow(composition), "spots,", k, "composition clusters.\n")
