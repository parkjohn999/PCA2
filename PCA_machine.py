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
    
    def r2PCA_cov(self, MP1, MP2, ret1, ret2):
        # MP means the moore penroses
        # Ret means return histories
        rel_rethist1 = np.matmul(ret1, MP1)
        ret_rethist2 = np.matmul(ret2, MP2)

        ret_rethist = np.concatenate((rel_rethist1, ret_rethist2), axis=1)

        df = pd.DataFrame(ret_rethist)

        return df.cov()
    
    def r2PCA_sceigs(self, cov_mat):
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        idx = eig_val.argsort()[::-1]
        print(eig_vec)
        eig_vec = eig_vec.T
        eig_val = eig_val[idx]
        eig_vec = eig_vec[idx]
        print("VALS")
        print(eig_val)
        print("VECS")
        print(eig_vec)
        final = np.sqrt(np.absolute(eig_val)) * eig_vec
        return final.T


    def components(self, scomps1, scomps2, sceigs):
        # Direct Sum scomps1 and scomps2
        dsum = np.zeros(np.add(scomps1.shape, scomps2.shape))
        dsum[:scomps1.shape[0],:scomps1.shape[1]]=scomps1
        dsum[scomps1.shape[0]:,scomps1.shape[1]:]=scomps2
        return np.matmul(sceigs, dsum)