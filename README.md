# Introduction
Single-cell RNA sequencing data are often characterized by sparsity, meaning than many data sets consists out of zeros for more than ninety percent of their total measurements. Until recently, this was considered to be the result of some technical error [1]. However, several studies suggest that this zero observation rate is biologically relevant [2]. This means that the dropout could potentially reveal information about the state of a cell, be a separating factor when considering cell types and generally reflect biological variation [1].  

In this study, three different transformation techniques on the gene expression are implemented. The first and second transformations that are applied on the gene expressions, process the transcriptome in such way so that it holds information about the zero rate observed. The third transformation used, takes into account the average expression of a gene across cells. The transformed gene expressions are then used as input to various automatic classifiers.

# Data availability

For this project we use PBMC data [3] as input.


# Project Structure

To begin with, run the main_data_setup.ipynb. This will make sure that the data used are binarized and compressed. 

Folders:

1. In folder `collective_runs`, you can find the python scripts used to run all classifiers for all data sets.

2. In folder `individual_runs`, you can find the python scripts used to run classification methods on the reference set and a test set you wish (from the available options) . Those scripts can be used to experiment with input parameters for the classifiers, transforming the data sets and trying new classifiers. For scripts in  `individual_runs`, more details are available (e.g. confusion matrices etc.) when using the classifiers.

3. In folder `classifiers`, you can find all classifiers used for the scripts found in the `collective_runs` folder.


4. In `ctc_utils/functions.py`, functions frequently used in this project can be found .

5. In folder `models`, trained classification models can be saved.

6. In folder `results`, results from the classifiers can be saved.

7. In folder `R_scripts`, you can find the classification experiments that use the SingleR classifier.

# References
[1]Valentine  Svensson.   Droplet  scrna-seq  is  not  zero-inflated.Nature Biotechnology 2020 38:2,38:147–150, 1 2020

[2] Gerard A. Bouland, Ahmed Mahfouz, and Marcel J.T. Reinders.  Differential dropout analysiscaptures biological variation in single-cell rna sequencing data.bioRxiv, 2021

[3]Jiarui  Ding,  Xian  Adiconis,  Sean  K.  Simmons,  Monika  S.  Kowalczyk,  Cynthia  C.  Hession,Nemanja D. Marjanovic, Travis K. Hughes, Marc H. Wadsworth, Tyler Burks, Lan T. Nguyen,John Y. H. Kwon, Boaz Barak, William Ge, Amanda J. Kedaigle, Shaina Carroll, Shuqiang Li,Nir  Hacohen,  Orit  Rozenblatt-Rosen,  Alex  K.  Shalek,  Alexandra-Chlo ́e  Villani,  Aviv  Regev,and Joshua Z. Levin.  Systematic comparative analysis of single cell rna-sequencing methods.bioRxiv, 2019.

