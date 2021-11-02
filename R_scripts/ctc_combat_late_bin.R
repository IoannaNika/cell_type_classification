### Load data ###
library(magrittr)
library(ggplot2)
library(reshape2)
library(caret)
library(SingleR)
# library(crfsuite)
library(sva)
library(crfsuite)
library(scAlign)
library(Seurat)
library(gridExtra)
library(harmony)

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
  
  #Seurat setup
  SeuratObj_train <- CreateSeuratObject(counts = x_train, project = "x_train", min.cells = 0)
  SeuratObj_train <- NormalizeData(SeuratObj_train)
  SeuratObj_train <- ScaleData(SeuratObj_train, do.scale=T, do.center=T, display.progress = T)
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
    
    
    #Seurat object setup for test set
    SeuratObj_test <- CreateSeuratObject(counts = x_test, project = "x_test", min.cells = 0)
    SeuratObj_test <- NormalizeData(SeuratObj_test)
    SeuratObj_test <- ScaleData(SeuratObj_test, do.scale=T, do.center=T, display.progress = T)
    SeuratObj_test@meta.data$type = y_test
    
    ## Gene selection
    SeuratObj_train <- FindVariableFeatures(SeuratObj_train, do.plot = F, nFeature=3000)
    SeuratObj_test<- FindVariableFeatures(SeuratObj_test, do.plot = F, nFeature=3000,)
    genes.use = Reduce(intersect, list(VariableFeatures(SeuratObj_train),
                                       VariableFeatures(SeuratObj_test),
                                       rownames(SeuratObj_train),
                                       rownames(SeuratObj_test)))
    
    # Save scaled train & test dataset
    scaled_train = SeuratObj_train@assays$RNA@scale.data[genes.use,]
    scaled_test = SeuratObj_test@assays$RNA@scale.data[genes.use,]
    
    
    # For batch correction using ComBat library
    l_train <- length(t(y_train))
    l_test <- length(t(y_test))
    l_trainpl <- l_train + 1
    total <- l_test+l_train

    batch <-rep(NA, total)
    batch[0:l_train] <- "train"
    batch[l_trainpl:total] <- "test"

    comb = cbind(scaled_train, scaled_test)
    
    combat_batched_corrected_data = ComBat(dat = comb, batch=batch, mod=NULL, par.prior=TRUE, prior.plots=FALSE)


    bc_train = combat_batched_corrected_data[,0:l_train]
    bc_test = combat_batched_corrected_data[,l_trainpl:total]


    # Binarisation 
    bc_train[bc_train>0] <-1
    bc_train[bc_train<0] <- 0
    bc_test[bc_test>0] <-1
    bc_test[bc_test<0] <-0
    
    # Case: data sets for train and testing are the same, in this case no batch correction is needed.
    if(i == j){
      
      #No need for batch correction in same datasets.
      bc_train = scaled_train
      
      bc_train[bc_train>0] <-1
      bc_train[bc_train<0] <- 0
      
      # Split training set into test set & training set 
      x_train_sc = bc_train
      x_test_sc = bc_train
      
      numOfCells = dim(x_train_sc)[2]
      amountForTest = as.integer(numOfCells*0.25)
      indx <-sample(0:numOfCells, amountForTest)
      
      bc_train = x_train_sc[,-indx]
      bc_test = x_test_sc[,indx]
      
      y_train = y_train[-indx,]
      y_test = y_test[indx,]
    }
    
    # Setting up the training:
    trained <- trainSingleR(bc_train, label= t(y_train))
    
    # Performing the classification:
    pred <- classifySingleR(bc_test, trained)
    
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
          key.title = ""
)



