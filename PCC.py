from preprocessing import *
import time


input_file = 'D:/neurb\OneDrive\python\GTA\data/MEScounts10426.h5ad'
adata = sc.read(input_file)
print(adata)
Genes = adata.var_names.values
Cells = adata.obs_names.values
express = pd.DataFrame(adata.X, columns=Genes, index=Cells)
print(express)
corr = express.corr()

start = time.time()

row, col = [], []
row_, col_ = [], []

for i in range(len(Genes)):
    for j in range(len(Genes)):
        if corr.iloc[i, j] > 0.3 and i != j:
            row.append(i)
            col.append(j)
        if corr.iloc[i, j] < -0.3 and i != j:
            row_.append(i)
            col_.append(j)

adj = sp.csc_matrix((np.ones(len(row)), (row, col)), shape=(len(Genes), len(Genes)))
adj_ = sp.csc_matrix((np.ones(len(row_)), (row_, col_)), shape=(len(Genes), len(Genes)))
end = time.time()
print(end-start)

sp.save_npz('./data/MEScounts_dataset_pos10426.npz', adj)
sp.save_npz('./data/MEScounts_dataset_neg10426.npz', adj_)