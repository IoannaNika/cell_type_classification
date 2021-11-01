from statistics import variance
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
import pickle


# Plots cell type number of occurances in the dataset.
# Input expected: labels of a set.
def plot_cell_type_counts(labels):
    import collections
    import matplotlib.pyplot as plt

    #count occurances per cell type
    label_counter_collection = collections.Counter(labels)

    counts = label_counter_collection.values()
    
    unique_labels = label_counter_collection.keys()
    
    #plot bar chart
    fig, ax = plt.subplots()
    ax.bar(unique_labels,counts, color = "purple")
    ax.tick_params('x', labelrotation=90)
    fig.tight_layout()
    plt.show()

# Takes the raw reads of gene expression
# Returns the gene expression binarized.
def binarize_cells_data(gene_expr):
    gene_expr_bin = np.copy(gene_expr)

    gene_expr_bin[gene_expr_bin > 0] = 1
    gene_expr_bin = gene_expr_bin.astype(np.bool)

    return gene_expr_bin

# returns binarized data, the cell names and the gene names
def get_cells_data(file_name):
    # load data
    df_data = pd.read_csv(file_name)

    # take the gene expression
    gene_expr_bin = df_data.iloc[:, 1:].to_numpy()

    gene_names = df_data.columns.values.tolist()[1:]
    cell_names = df_data.iloc[:, 0].tolist()

    return (gene_expr_bin, cell_names, gene_names)

# Get target from file  
def get_labels(file_name):
    return pd.read_csv(file_name).iloc[:, 0].values.tolist()

# Unpickles files & decompresses gene expression
def load_pickled_cells_data(
    cell_names_file_name,
    gene_names_file_name,
    labels_file_name,
    gene_expr_bin_bitpacked_file_name,
):
    cell_names = pickle.load(open(cell_names_file_name, "rb"))
    gene_names = pickle.load(open(gene_names_file_name, "rb"))
    labels = pickle.load(open(labels_file_name, "rb"))

    gene_expr_bin_bitpacked = np.load(gene_expr_bin_bitpacked_file_name)
    gene_expr_bin = np.unpackbits(
        gene_expr_bin_bitpacked, axis=-1, count=len(gene_names)
    )

    return (cell_names, gene_names, labels, gene_expr_bin)

# Compression with no binarization of gene expression
def write_pickled_cell_data_nb(
    cells_labels_file_name,
    cell_names_file_name,
    gene_names_file_name,
    labels_file_name,
    gene_expr_bin_bitpacked_file_name,
    gene_expr_bin, 
    cell_names, 
    gene_names
):
    # Get labels
    labels = get_labels(cells_labels_file_name)

    # Store files as binary files - to save space/time - 
    pickle.dump(cell_names, open(cell_names_file_name, "wb"))
    pickle.dump(gene_names, open(gene_names_file_name, "wb"))
    pickle.dump(labels, open(labels_file_name, "wb"))

    #compression of binarized gene expression 
    np.save(gene_expr_bin_bitpacked_file_name, np.packbits(gene_expr_bin, axis=-1))

# Compression & binarization of gene expression
def write_pickled_cell_data(
    cells_data_file_name,
    cells_labels_file_name,
    cell_names_file_name,
    gene_names_file_name,
    labels_file_name,
    gene_expr_bin_bitpacked_file_name,
):

    # Split data from file into gene expr, cell_names and gene names
    (gene_expr, cell_names, gene_names) = get_cells_data(cells_data_file_name)

    # Get labels
    labels = get_labels(cells_labels_file_name)

    # Store files as binary files - to save space/time - 
    pickle.dump(cell_names, open(cell_names_file_name, "wb"))
    pickle.dump(gene_names, open(gene_names_file_name, "wb"))
    pickle.dump(labels, open(labels_file_name, "wb"))
    
    # Binarize data
    gene_expr_bin = binarize_cells_data(gene_expr)

    #compression of binarized gene expression 
    np.save(gene_expr_bin_bitpacked_file_name, np.packbits(gene_expr_bin, axis=-1))

# Calculates dlr for a specific cell
###############
# Input: 
# -------
# 1. Indices of the neighbors (includes the sample itself)
# 2. Reference set
# ############### 
# Output:
# -------
# 1.  DLR for a specific cell.
def calcDLR(neighbors, gene_expr_bin):
    #exludes from the dataset the sample and the neighbors
    exclDataset = np.delete(gene_expr_bin, neighbors,0)
    #will save actual gene expressions of neighbours - neighbors includes the sample as well-. 
    actual_neighbours = gene_expr_bin[neighbors, :]

    sum_in = np.sum(actual_neighbours, axis = 0)

    sum_out = np.sum(exclDataset, axis = 0)

    r_in = sum_in/len(actual_neighbours[:, 0])

    r_out = sum_out/len(exclDataset[:, 0])

    division = np.divide(r_in, (r_out+1))
    
    log_div = np.log(division+1) 
    
    return log_div

# Calculates dlr for all cells.
###############
# Input: 
# -------
# 1. Indices of the neighbors (includes the sample itself)
# 2. Reference set
###############
# Output:
# -------
# 1.  DLR matrix.
def loopForDLR(neighbors_index_ps, x_train):
    for row in range(len(neighbors_index_ps)):
        neighbors_index = neighbors_index_ps[row]
        dlr = calcDLR(neighbors_index, x_train)
        if row == 0:
            matrix = dlr
        else:
            matrix = np.vstack((matrix, dlr)) 
    return matrix


# Unsupervised clustering using the Faiss library
###############
# Input:
# ------
# 1. x_dt_fs : training set after feature selection has been applied (cells x genes)
# 2. x_dt_test_fs : test set after feature selection has been applied (cells x genes)
# 3. n : number of neighbours to consider.
###############
# Output
# -------
# 1. neighbors_index_ps : indices of the neighbours of cells (taken from the train dataset) in the training dataset 
# 2. neighbors_index_ps : indices of the neighbours of cells (taken from the test dataset) in the training dataset
# 3. x_dt_fs : training set after feature selection has been applied (cells x genes)
# 4. x_dt_test_fs : test set after feature selection has been applied (cells x genes)
def unsupervisedClusteringFaiss(x_dt_fs, x_dt_test_fs, n):
    import faiss
    import numpy as np
    # Unsupervised clustering using faiss (Euclidean distance)
    index = faiss.IndexFlatL2(len(x_dt_fs[0])) 
    # Setup for the training dataset
    x_dt_fs = np.ascontiguousarray(x_dt_fs)
    x_dt_fs = x_dt_fs.astype('float32')
    # Train and populate model
    index.train(x_dt_fs)
    index.add(x_dt_fs)

    # setup for the testing dataset 
    x_dt_test_fs = np.ascontiguousarray(x_dt_test_fs)
    x_dt_test_fs = x_dt_test_fs.astype('float32')
    
    #number of neighbors
    num = n

    distances, neighbors_index_ps = index.search(x_dt_fs, num)
    distances_test, neighbors_index_ps_test = index.search(x_dt_test_fs, num)

    return neighbors_index_ps, neighbors_index_ps_test, x_dt_fs, x_dt_test_fs 

# Loads stored dlr data (in case they were pre computed) 
def load_data():
    cell_names_file_name = "../data/cell_names"
    gene_names_file_name = "../data/gene_names"
    labels_file_name = "../data/labels"
    gene_expr_bin_bitpacked_file_name = "../data/gene_expr_bin_bitpacked"

    (cell_names, gene_names, labels, gene_expr_bin) = load_pickled_cells_data(
    cell_names_file_name,
    gene_names_file_name,
    labels_file_name,
    gene_expr_bin_bitpacked_file_name + ".npy",)
    
    return cell_names, gene_names, labels, gene_expr_bin

# Divide gene expression into corresponding dataset
def def_gen_expr_for_datasets(gene_expr_bin):
    ge_10xv2 = gene_expr_bin[0:6443]
    ge_SM2 = gene_expr_bin[6444:6696]
    ge_10xv3 = gene_expr_bin[6697:9918]
    ge_CL = gene_expr_bin[9919:10171]
    ge_DR = gene_expr_bin[10172:13393]
    ge_iD = gene_expr_bin[13394:16615]
    ge_SW = gene_expr_bin[16616:19791]
    ge_10xv2_2 = gene_expr_bin[19792:23153]

    return ge_10xv2, ge_SM2, ge_10xv3, ge_CL, ge_DR, ge_iD, ge_SW, ge_10xv2_2

# Divide labels into corresponding dataset
def def_labels_for_datasets(labels):
    lb_10xv2 = labels[0:6443]
    lb_SM2 = labels[6444:6696]
    lb_10xv3 = labels[6697:9918]
    lb_CL = labels[9919:10171]
    lb_DR = labels[10172:13393]
    lb_iD = labels[13394:16615]
    lb_SW = labels[16616:19791]
    lb_10xv2_2 = labels[19792:23153]

    return lb_10xv2, lb_SM2, lb_10xv3, lb_CL, lb_DR, lb_iD, lb_SW, lb_10xv2_2

# Removes cells & the correspinging labels so that the total amount of cells matches the threshold.
def prune_training_set(gene_expr_bin, labels, threshold):
    from collections import  Counter
    import numpy as np
    d = Counter(labels)
    min_occurances = min(d, key=d. get)
    min_num = d.get(min_occurances)

    remove_indices = np.array([])

    for key in d:
        indices = [i for i, x in enumerate(labels) if x == key]
        if len(indices)> threshold: #3158 #3711
            ind = indices[threshold:len(indices)]
            remove_indices = np.append(remove_indices, ind)
        
    new_labels = [v for i, v in enumerate(labels) if i not in remove_indices]
    new_gene_expression = [v for i, v in enumerate(gene_expr_bin) if i not in remove_indices]

    y = new_labels
    x = np.array(new_gene_expression)

    return x, y

# Plots cluster contents  
def plot_cluster_contents(linkage_matrix, y, clusterNumber):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import ward, fcluster
    from scipy.spatial.distance import pdist

    res = fcluster(linkage_matrix, t=clusterNumber, criterion='maxclust')

    celltypes_occurences_per_cluster = []

    for i in range(clusterNumber):
        celltypes_occurences_per_cluster.append(dict((el,0) for el in set(y)))

    for index, value in enumerate(res):
        celltypes_occurences_per_cluster[value-1][y[index]] += 1

    fig, ax = plt.subplots()

    lbl = list(range(clusterNumber))

    counts_per_cluster = []

    for cell_type in set(y):
        occurences_per_cluster_current_cell = []

        for i in range(clusterNumber):
            occurences_per_cluster_current_cell.append(celltypes_occurences_per_cluster[i][cell_type])

        counts_per_cluster.append(occurences_per_cluster_current_cell)

        ax.plot(lbl, occurences_per_cluster_current_cell, label=cell_type)

    counts_per_cluster = np.array(counts_per_cluster)

    print(np.sum(counts_per_cluster, axis=0))

    ax.vlines(lbl, 0, 3000, color="black", linestyles = "dashed")
    fig.set_size_inches(18.5, 10.5)
    ax.legend()
    plt.show()