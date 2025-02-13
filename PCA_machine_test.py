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

print(MPCO)