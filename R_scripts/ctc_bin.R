### Load data ###
library(magrittr)
library(ggplot2)
library(reshape2)
library(caret)
library(SingleR)
library(crfsuite)
library(sva)
library(crfsuite)
library(scAlign)
library(Seurat)
library(SingleCellExperiment)
library(gridExtra)

results.dir = "results"

### Load data ###
data <- rio::import("data/10Xv2_pbmc1.csv")
rownames(data) <- data$V1
data$V1 <- NULL
data <- t(data)
idx <- list(1:6444,
            6445:6697,
            6698:9919,
            9920:10172,
            10173:13394,
            13395:16616,
            16617:19792,
            19793:23154)

names(idx) <- c("10xv2","SM2","10xv3","CL","DR","iD","SW","10xv2_2")
labels <- read.csv("Data/10Xv2_pbmc1Labels.csv")
colnames(labels) <- "celltype"

# genes x cells
datasets <- lapply(idx,function(x){
  colnames <- colnames(data[,x])
  celltype <- labels[x,"celltype"]
  samplesheet <- data.frame("colnames" = colnames,
                            "celltype" = celltype)
  return(list(data[,x],samplesheet))
})

# Define results matrix
results = matrix(0, length(datasets), length(datasets))

i<-0
for(dt_train in datasets){
  #increment i
  i<-i+1
  
  # setup for training set and labels
  x_train = dt_train[[1]]
  y_train = dt_train[[2]]["celltype"]
  
  # binarization
  x_train[x_train>0] <- 1
  
  #Seurat object setup for train set
  SeuratObj_train <- CreateSeuratObject(counts = x_train, project = "x_train", min.cells = 0)
  SeuratObj_train@meta.data$type = y_train
  
  j<-0
  for(dt_test in datasets){
    #increment j
    j<-j+1
    
    # In case it got changed when i == j 
    y_train = dt_train[[2]]["celltype"]
    
    # Setup test set and labels
    x_test = dt_test[[1]]
    y_test = dt_test[[2]]["celltype"]
    
    # binarization
    x_test[x_test>0] <- 1
    
    #Seurat object setup for test set
    SeuratObj_test <- CreateSeuratObject(counts = x_test, project = "x_test", min.cells = 0)
    SeuratObj_test@meta.data$type = y_test
    
    ## Gene selection
    SeuratObj_train <- FindVariableFeatures(SeuratObj_train, do.plot = F, nFeature=3000)
    SeuratObj_test<- FindVariableFeatures(SeuratObj_test, do.plot = F, nFeature=3000,)
    genes.use = Reduce(intersect, list(VariableFeatures(SeuratObj_train),
                                       VariableFeatures(SeuratObj_test),
                                       rownames(SeuratObj_train),
                                       rownames(SeuratObj_test)))
    
    # Save scaled train & test dataset
    fs_train = SeuratObj_train@assays$RNA@counts[genes.use,]
    fs_test = SeuratObj_test@assays$RNA@counts[genes.use,]
    
    # Binarisation 
    fs_train[fs_train>0] <-1
    fs_test[fs_test>0] <-1

    
    # Case: data sets for train and testing are the same
    if(i == j){
      
      # Split training set into test set & training set 
      x_train_sc = fs_train
      x_test_sc = fs_train
      
      numOfCells = dim(x_train_sc)[2]
      amountForTest = as.integer(numOfCells*0.25)
      indx <-sample(0:numOfCells, amountForTest)
      
      fs_train = x_train_sc[,-indx]
      fs_test = x_test_sc[,indx]
      
      y_train = y_train[-indx,]
      y_test = y_test[indx,]
    }
    
    # Setting up the training:
    trained <- trainSingleR(fs_train, label= t(y_train))
    
    # Performing the classification:
    pred <- classifySingleR(fs_test, trained)
    
    #Calculate weighted f1
    table(predicted=pred$labels, truth= t(y_test))
    metrics <-crf_evaluation(
      pred$labels,
      t(y_test)
    )
    
    f1_metrics = metrics["bylabel"][[1]]["f1"]
    support = metrics["bylabel"][[1]]["support"]
    total_sum = sum(support)
    weighted_f1 = (f1_metrics * support)
    weighted_f1[is.na(weighted_f1)] <- 0
    weighted_f1 = sum(weighted_f1)/total_sum
    print(weighted_f1)
    results[i,j] = weighted_f1
    
  }
  
}

#Plotting
colnames(results) <- names(datasets)
rownames(results)<- names(datasets)
library("gplots")
library("RColorBrewer")
heatmap.2(results, Colv = NA, Rowv = NA,col=brewer.pal(11,"RdBu"),
          dendogram = "none", trace="none", density.info='none',
          cellnote = round(results, 2), notecol="black",
          margins = c(6, 6),
          ylab = "Training set",
          xlab = "Test set",
          key.xlab = "Weighted f1 score",
          key.title = "",
          key = TRUE
)



