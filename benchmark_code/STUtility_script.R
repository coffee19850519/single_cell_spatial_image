source("/scratch/10x/Yuzhou_script/read_npz.R")
library(STutility)
library(Seurat)
library(optparse)
option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = "/scratch/10x/Human_breast_cancer/", 
              help = "input path", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = "/scratch/10x/Human_brain_JH_12/result_new/STUtility_default/",
              help = "output path", metavar = "character"),
  make_option(c("-s","--sample"),type = "character",default = "151507",
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
              help = "prefix of output files: combination and default", metavar = "character")
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

STutility_func <- function(input_dir = input_dir,
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
  my.seurat.object <- Seurat::CreateSeuratObject(mat_in)
  my.seurat.object <- Seurat::FindVariableFeatures(my.seurat.object,verbose = F)
  my.seurat.object <- Seurat::ScaleData(my.seurat.object,verbose = F)
  my.seurat.object <- STutility::RunNMF(my.seurat.object, nfactors = PCs)
  my.seurat.object <- Seurat::FindNeighbors(object = my.seurat.object, verbose = FALSE, reduction = "NMF", dims = 1:PCs)
  my.seurat.object <- Seurat::FindClusters(object = my.seurat.object, verbose = FALSE)
  results <- as.data.frame(cbind(barcode = names(my.seurat.object$seurat_clusters),
                                 label = as.character(my.seurat.object$seurat_clusters)))
  if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
  if(prefix == "default") {parameters <- "default"}
  save_name <- paste0(samples, "_STUtility_", normalization_method, "_", parameters, ".csv")
  write.csv(results, file = file.path(save_dir, save_name), row.names = FALSE)
}

time.p1 <- Sys.time()
p <- profmem::profmem(STutility_func(input_dir = input_dir,
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
save_name <- paste0(samples, "_STUtility_", normalization_method, "_", parameters, ".csv")
run_log<- paste0("success at:", save_name, ";mem:",mem_total, ";time:",time_cost) #},
#                   warning = function(w){
#                     if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
#                     if(prefix == "default") {parameters <- "default"}
#                     save_name <- paste0(samples, "_STUtility_", normalization_method, "_", parameters, ".csv")
#                     return(paste0("warning at:", save_name))},
#                   error = function(e){
#                     if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
#                     if(prefix == "default") {parameters <- "default"}
#                     save_name <- paste0(samples, "_STUtility_", normalization_method, "_", parameters, ".csv")
#                     return(paste0("error at:", save_name))
#                   })
if(!file.exists(file.path(output_dir,"STUtility_log","STUtility_log"))){
  dir.create(file.path(output_dir,"STUtility_log"))
  file.create(file.path(output_dir,"STUtility_log","STUtility_log"))
}

write(run_log,file.path(output_dir,"STUtility_log","STUtility_log"),append = T)

