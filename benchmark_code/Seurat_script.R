source("/scratch/10x/Yuzhou_script/read_npz.R")
library(Seurat)
library(optparse)
option_list <- list(
  make_option(c("-i", "--input"), type = "character", 
              default = "/scratch/10x/Simulation/downsampled_matrices/cell_simulation/", 
              help = "input path", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", 
              default = "/scratch/10x/Human_brain_JH_12/result_new/Seurat_simulation/",
              help = "output path", metavar = "character"),
  make_option(c("-s","--sample"),type = "character",default = "lvl1.counts.151673",
              help = "input sample name", metavar = "character"),
  make_option(c("-d","--normalization"), type = "character", default = "LogCPM",
              help = "input normalization dir", metavar = "character"),
  make_option(c("-p","--PCs"), type = "numeric", default = 128,
              help = "number of PCs", metavar = "numeric"),
  make_option(c("-n","--neighbor"), type = "numeric", default = 20,
              help = "number of neighbors", metavar = "numeric"),
  make_option(c("-r","--resolution"), type = "numeric", default = 0.8,
              help = "resolution for clustering", metavar = "numeric"),
  make_option(c("-f","--prefix"), type = "character",default = "default",
              help = "prefix of output files: combination and default", 
              metavar = "character")
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)
# input parameter
input_dir <- opt$input
output_dir <- opt$output
samples <- opt$sample
normalization_method <- opt$normalization
PCs <- opt$PCs
neighbor <- opt$neighbor
resolution <- opt$resolution
prefix <- opt$prefix
# list all dir

Seurat_func <- function(input_dir = input_dir,
                        output_dir = output_dir,
                        samples = samples,
                        normalization_method = normalization_method,
                        PCs = PCs,
                        neighbor = neighbor,
                        resolution = resolution,
                        prefix = prefix)
  {
  all_sample_expression_dir <- file.path(input_dir,"normalized_data",normalization_method,"spa_out")
  all_sample_meta_dir <- file.path(input_dir,"sparse_meta_out")
  # list all files in dir
  all_files_npz <- list.files(all_sample_expression_dir,pattern = "npz")
  all_file_meta <- list.files(all_sample_meta_dir,pattern = "meta")
  all_file_genename <- list.files(all_sample_expression_dir,pattern = "gene")
  # 
  one_npz_path <- file.path(all_sample_expression_dir, 
                            grep(samples,all_files_npz,value = T,ignore.case = T))  
  one_gene_path <- file.path(all_sample_expression_dir,
                             grep(samples,all_file_genename,value = T,ignore.case = T))
  one_spot_path <- file.path(all_sample_meta_dir,
                             grep(samples, all_file_meta, value = T, ignore.case = T))
  save_dir <- output_dir
  mat_in <- npz_read(one_npz_path,one_gene_path,one_spot_path)
  mydata <- Seurat::CreateSeuratObject(counts = mat_in, assay = 'Spatial')
  
  mydata <- Seurat::FindVariableFeatures(mydata)
  mydata <- Seurat::ScaleData(mydata)
  mydata <- Seurat::RunPCA(mydata, assay = "Spatial", npcs = PCs)
  mydata <- Seurat::FindNeighbors( mydata,k.param = neighbor, reduction = "pca",dims = 1:PCs)
  mydata <- Seurat::FindClusters(mydata,resolution = resolution)
  results <- as.data.frame(mydata@active.ident)
  results$barcode <- rownames(results)
  colnames(results) <- c("label", "barcode")
  results <- results[, c("barcode", "label")]
  if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
  if(prefix == "default") {parameters <- "default"}
  save_name <- paste0(samples, "_Seurat_", normalization_method, "_", parameters, ".csv")
  write.csv(results, file = file.path(save_dir, save_name), row.names = FALSE)
  }

run_log <- tryCatch({
  time.p1 <- Sys.time()
  p <- profmem::profmem(Seurat_func(input_dir = input_dir,
                                 output_dir = output_dir,
                                 samples = samples,
                                 normalization_method = normalization_method,
                                 PCs = PCs,
                                 neighbor = neighbor,
                                 resolution = resolution,
                                 prefix = prefix));
  time.p2 <- Sys.time()
  time_cost <- difftime(time.p2, time.p1, units = "secs")
  mem_total <- max(p$bytes,na.rm = T)
  if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
  if(prefix == "default") {parameters <- "default"}
  save_name <- paste0(samples, "_Seurat_", normalization_method, "_", parameters, ".csv")
  paste0("success at:", save_name, ";mem:",mem_total, ";time:",time_cost)},
                    warning = function(w){
                      time.p1 <- Sys.time()
                      p <- profmem::profmem(Seurat_func(input_dir = input_dir,
                                                        output_dir = output_dir,
                                                        samples = samples,
                                                        normalization_method = normalization_method,
                                                        PCs = PCs,
                                                        neighbor = neighbor,
                                                        resolution = resolution,
                                                        prefix = prefix));
                      time.p2 <- Sys.time()
                      time_cost <- difftime(time.p2, time.p1, units = "secs")
                      mem_total <- max(p$bytes,na.rm = T)
                      if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
                      if(prefix == "default") {parameters <- "default"}
                      save_name <- paste0(samples, "_Seurat_", normalization_method, "_", parameters, ".csv")
                      return(paste0("warnings at:", save_name, ";mem:",mem_total, ";time:",time_cost))},
                    error = function(e){
                      if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
                      if(prefix == "default") {parameters <- "default"}
                      save_name <- paste0(samples, "_Seurat_", normalization_method, "_", parameters, ".csv")
                      return(paste0("error at:", save_name))
                    })
if(!file.exists(file.path(output_dir,"Seurat_log","Seurat_log"))){
  dir.create(file.path(output_dir,"Seurat_log"))
  file.create(file.path(output_dir,"Seurat_log","Seurat_log"))
}

write(run_log,file.path(output_dir,"Seurat_log","Seurat_log"),append = T)

