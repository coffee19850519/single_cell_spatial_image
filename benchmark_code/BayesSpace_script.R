source("/scratch/10x/Yuzhou_script/read_npz.R")
library(BayesSpace)
library(Seurat)
library(optparse)
library(SingleCellExperiment)

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = "/scratch/10x/Human_breast_cancer/", 
              help = "input path", metavar = "character"),
  make_option(c("-o", "--output"), type = "character", default = "/scratch/10x/Human_brain_JH_12/result_new/BayesSpace_default/",
              help = "output path", metavar = "character"),
  make_option(c("-s","--sample"),type = "character",default = "1142243F",
              help = "input sample name", metavar = "character"),
  make_option(c("-d","--normalization"), type = "character", default = "LogCPM",
              help = "input normalization dir", metavar = "character"),
  make_option(c("-p","--PCs"), type = "numeric", default = 128,
              help = "number of PCs", metavar = "numeric"),
  make_option(c("-q","--clusters"), type = "numeric", default = 4,
              help = "number of int cluster", metavar = "numeric"),
  make_option(c("-r","--iteration"), type = "numeric", default = 50000,
              help = "number of iteration", metavar = "numeric"),
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
int.cluster <- opt$clusters
iteration.num <- opt$iteration
prefix <- opt$prefix
# list all dir

Bayes_func <- function(input_dir = input_dir,
                           output_dir = output_dir,
                           samples = samples,
                           normalization_method = normalization_method,
                           PCs = PCs,
                           int.cluster = int.cluster,
                           iteration.num = iteration.num,
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
  colData <- read.csv(one_spot_path, header = TRUE)
  colData <- colData[, 1:6]
  colnames(colData) <- c("spot", "in_tissue", "row", 
                         "col", "imagerow", "imagecol")
  rownames(colData) <- colData$spot
  
  # create SingleCellExperiment object
  sce <- SingleCellExperiment(assays=list(counts=mat_in),
                              colData=colData)
  
  # process avoiding normalization and cluster
  set.seed(102)
  sce <- spatialPreprocess(sce, platform="Visium", 
                           n.PCs=PCs,n.HVGs = 2000, log.normalize=FALSE, assay.type = 'counts')
  sce <- qTune(sce, platform="Visium")
  sce <- spatialCluster(sce, q= int.cluster,nrep = iteration.num, init.method = "kmeans", model="t", platform = "Visium")
  # sce.enhanced <- spatialEnhance(sce, q=7, platform="Visium", d=7,
  #                                     model="t", gamma=2,
  #                                     jitter_prior=0.3, jitter_scale=3.5,
  #                                     nrep=1000, burn.in=100,
  #                                     save.chain=TRUE)
  # 
  
  # obtain results and save
  results <- as.data.frame(cbind(barcode = sce$spot, label = sce$spatial.cluster))
  colnames(results) <- c("barcode", "label")
  # comments here  start
  # my.coldata <- colData(sce.enhanced)
  # spot.idx <- my.coldata$spot.idx
  # spot.barcode <- rownames(colData(sce))[spot.idx] 
  # my.coldata$spot.barcode <- spot.barcode  
  # here *******************************
  if(prefix == "combination"){parameters <- paste0("IntCluster-", int.cluster, "-npc-", PCs, "-iter-", iteration.num)}
  if(prefix == "default") {parameters <- "default"}
  # comments here  start
  # save_name <- paste0(samples, "_BayesSpace_", normalization_method, "_", parameters, "_enhanced_C7.csv") # warning: for other parameters, the save_name should be changed
  # write.csv(my.coldata, file = paste0(save_dir, save_name), row.names = FALSE)
  # here *******************************
  save_name <- paste0(samples, "_BayesSpace_", normalization_method, "_", parameters, ".csv") # warning: for other parameters, the save_name should be changed
  write.csv(results, file = paste0(save_dir, save_name), row.names = FALSE)
  #save(sce, file = paste0(save_dir, save_name,".RData"))
}

time.p1 <- Sys.time()
p <- profmem::profmem(Bayes_func(input_dir = input_dir,
                                     output_dir = output_dir,
                                     samples = samples,
                                     normalization_method = normalization_method,
                                     PCs = PCs,
                                     int.cluster = int.cluster,
                                     iteration.num = iteration.num,
                                     prefix = prefix));
time.p2 <- Sys.time()
time_cost <- difftime(time.p2, time.p1, units = "secs")
mem_total <- max(p$bytes,na.rm = T)
if(prefix == "combination"){parameters <- paste0("IntCluster-", int.cluster, "-npc-", PCs, "-iter-", iteration.num)}
if(prefix == "default") {parameters <- "default"}
save_name <- paste0(samples, "_BayesSpace_", normalization_method, "_", parameters, ".csv")
run_log<- paste0("success at:", save_name, ";mem:",mem_total, ";time:",time_cost) #},
#                   warning = function(w){
#                     if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
#                     if(prefix == "default") {parameters <- "default"}
#                     save_name <- paste0(samples, "_BayesSpace_", normalization_method, "_", parameters, ".csv")
#                     return(paste0("warning at:", save_name))},
#                   error = function(e){
#                     if(prefix == "combination"){parameters <- paste0("res-", resolution, "-npc-", PCs, "-neighbors-", neighbor)}
#                     if(prefix == "default") {parameters <- "default"}
#                     save_name <- paste0(samples, "_BayesSpace_", normalization_method, "_", parameters, ".csv")
#                     return(paste0("error at:", save_name))
#                   })
if(!file.exists(file.path(output_dir,"BayesSpace_log","BayesSpace_log"))){
  dir.create(file.path(output_dir,"BayesSpace_log"))
  file.create(file.path(output_dir,"BayesSpace_log","BayesSpace_log"))
}

write(run_log,file.path(output_dir,"BayesSpace_log","BayesSpace_log"),append = T)

