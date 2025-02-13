import numpy as np
import pandas as pd

class PCA_machine:
    def __init__(self, df):
        self.df = df

    def vol_calc(self, col):
        mean = np.mean(col)
        var = np.sum(np.power(col - mean,2)) / col.shape
        return np.sqrt(var)[0]

    def scaled_eigenvectors(self, mat):
        cov_mat = mat.cov()
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        return np.sqrt(np.absolute(eig_val)) * eig_vec
    
    def moorePenrose(self, mat):
        trans = np.transpose(mat)
        return -1*np.matmul(trans, np.linalg.inv(np.matmul(mat, trans)))
    
    def r2PCA(self, MP1, MP2, ret1, ret2):
        # MP means the moore penroses
        # Ret means return histories
        rel_rethist1 = np.matmul(ret1, MP1)
        ret_rethist2 = np.matmul(ret2, MP2)

        ret_rethist = np.concatenate((rel_rethist1, ret_rethist2), axis=1)

        return ret_rethist.cov()