import pandas as pd
from os import listdir
from os.path import isfile, join
mypath="/home/sohom/Desktop/AV_IITBHU_Enigma_Sep17/output_files/"
only_csv_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:]==".csv"]

df_all=pd.concat([pd.read_csv(mypath+fl) for fl in only_csv_files])
ensembled_ans=df_all.groupby('ID',as_index=False)['attempts_range'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("ebsemble1.csv",index=False)
