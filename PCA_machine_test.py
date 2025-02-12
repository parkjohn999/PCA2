from PCA_machine import *
from Data import *
from sklearn.decomposition import PCA

def var_table(df, PCA):
    vars = []
    for col in df.columns:
        vars.append([col, PCA.vol_calc(df[col])])

    return vars

df = Data().get_log_rets()

PCAm = PCA_machine(df)
df_CO = df[['CO1', 'CO6', 'CO12', 'CO18', 'CO24']]

print(df_CO)
#print(PCAm.scaled_eigenvectors(df_CO)*1000)