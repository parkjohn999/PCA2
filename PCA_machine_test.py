from PCA_machine import *
from Data import *
from sklearn.decomposition import PCA
import numpy as np

def var_table(df, PCA):
    vars = []
    for col in df.columns:
        vars.append([col, PCA.vol_calc(df[col])])

    return vars

df = Data().get_log_rets()

PCAm = PCA_machine(df)
df_CO = df[['CO1', 'CO6', 'CO12', 'CO18', 'CO24']]
df_QS = df[['QS1', 'QS6', 'QS12', 'QS18', 'QS24']]

#print(PCAm.scaled_eigenvectors(df_QS)*1000)

Csceigs = PCAm.scaled_eigenvectors(df_CO)
Qsceigs = PCAm.scaled_eigenvectors(df_QS)

selected_Csceigs = np.transpose(Csceigs[:, :2])
selected_Qsceigs = np.transpose(Qsceigs[:, :2])

MPCO = PCAm.moorePenrose(selected_Csceigs)
MPQS = PCAm.moorePenrose(selected_Qsceigs)

cov = PCAm.r2PCA_cov(MPCO, MPQS, df_CO, df_QS)

#sceigs = PCAm.r2PCA_sceigs(cov)

test_mat = np.array([[1, 0, 0.745, -0.075],
                    [0,1,0.031,0.285],
                    [0.745, 0.031, 1, 0],
                    [-0.075, 0.285, 0, 1]])

print(PCAm.r2PCA_sceigs(test_mat))

#print(1000*PCAm.components(selected_Csceigs, selected_Qsceigs, sceigs))
