import pandas as pd
rnfl=pd.read_csv("rnfl.csv")

#total average is the average of OD and OS
rnfl["Total Avgs"]=(rnfl["Average OD"]+rnfl["Average OS"])/2

# creating separate DataFrames for each diagnosis
controls=rnfl.drop(rnfl[rnfl["Diagnosis"]!="Control"].index,inplace=False)

rbd_paf=rnfl.drop(rnfl[rnfl["Diagnosis"]!="RBD_PAF"].index,inplace=False)

msa=rnfl.drop(rnfl[rnfl["Diagnosis"]!="MSA"].index,inplace=False)

# find the n of each diagnosis
len(controls)
len(rbd_paf)
len(msa)

# n% of males
nmcontrols=(len(controls.loc[controls["Gender"]=="Male"])/len(controls))*100

nmrbd_paf=(len(rbd_paf.loc[rbd_paf["Gender"]=="Male"])/len(rbd_paf))*100

nmmsa=(len(msa.loc[msa["Gender"]=="Male"])/len(msa))*100

# avg and std age
avg_age_controls=controls["Age"].mean()

std_age_controls=controls["Age"].std()

avg_age_rbdpaf=rbd_paf["Age"].mean()

std_age_rbdpaf=rbd_paf["Age"].std()

avg_age_msa=msa["Age"].mean()

std_age_msa=msa["Age"].std()

# avg disease duration - not able to calculate without year symptoms began or date of diagnosis


# mean rnfl
avg_rnfl_controls=controls["Total Avgs"].mean()

avg_rnfl_rbdpaf=rbd_paf["Total Avgs"].mean()

avg_rnfl_msa=msa["Total Avgs"].mean()

#median rnfl
med_rnfl_controls=controls["Total Avgs"].median()

med_rnfl_rbdpaf=rbd_paf["Total Avgs"].median()

med_rnfl_msa=msa["Total Avgs"].median()

#min rnfl
min_rnfl_controls=controls["Total Avgs"].min()

min_rnfl_rbdpaf=rbd_paf["Total Avgs"].min()

min_rnfl_msa=msa["Total Avgs"].min()

# max rnfl
max_rnfl_controls=controls["Total Avgs"].max()

max_rnfl_rbdpaf=rbd_paf["Total Avgs"].max()

max_rnfl_msa=msa["Total Avgs"].max()

# range rnfl
range_rnfl_controls=max_rnfl_controls-min_rnfl_controls

range_rnfl_rbdpaf=max_rnfl_rbdpaf-min_rnfl_rbdpaf

range_rnfl_msa=max_rnfl_msa-min_rnfl_msa
