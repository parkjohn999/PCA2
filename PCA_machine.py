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
        print(cov_mat*1000)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        return np.sqrt(np.absolute(eig_val)) * eig_vec