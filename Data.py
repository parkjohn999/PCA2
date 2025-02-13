import glob
import pandas as pd
import numpy as np
import datetime
import platform

class Data:
    START_DATE = datetime.date(2011,9,26)
    END_DATE = datetime.date(2021,9,24)

    def __init__(self):
        glued = pd.DataFrame()
        for file_name in glob.glob('./Data/*.csv'):
            df = pd.read_csv(file_name, index_col="Date", usecols=[0,1])
            col_name = ''
            if 'Windows' in platform.platform():
                col_name = file_name.split('/')[1].split('\\')[1].split('.')[0]
            else:
                col_name = file_name.split('/')[2].split('.')[0]
            df = df.rename(columns={"Last Price":col_name})
            glued = pd.concat([glued, df], axis=1)
        self.df = glued[::-1]
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df[self.START_DATE:self.END_DATE]
        self.returns = self.df.dropna().pct_change().dropna()
        self.log_rets = np.log(1+self.returns).dropna()
        
    def get_data(self):
        return self.df
    
    def get_log_rets(self):
        return self.log_rets
    
    def get_rets(self):
        return self.returns

    