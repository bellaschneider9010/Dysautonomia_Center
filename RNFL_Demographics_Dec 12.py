import pandas as pd
rnfl=pd.read_csv("rnfl.csv")

#total average is the average of OD and OS
rnfl["Total Avgs"]=(rnfl["Average OD"]+rnfl["Average OS"])/2

# creating separate DataFrames for each diagnosis
controls=rnfl.drop(rnfl[rnfl["Diagnosis"]!="Control"].index,inplace=False)

rbd_paf=rnfl.drop(rnfl[rnfl["Diagnosis"]!="RBD_PAF"].index,inplace=False)

msa=rnfl.drop(rnfl[rnfl["Diagnosis"]!="MSA"].index,inplace=False)

# find the n of each diagnosis
print(len(controls), len(rbd_paf), len(msa))

# n% of males
nmcontrols=len(controls.loc[controls["Gender"]=="Male"])
print(nmcontrols/len(controls)*100)

nmrbd_paf=len(rbd_paf.loc[rbd_paf["Gender"]=="Male"])
print(nmrbd_paf/len(rbd_paf)*100)

nmmsa=len(msa.loc[msa["Gender"]=="Male"])
print(nmmsa/len(msa)*100)

# avg and std age
avg_age_controls=controls["Age"].mean()
print(avg_age_controls)

std_age_controls=controls["Age"].std()
print(std_age_controls)

avg_age_rbdpaf=rbd_paf["Age"].mean()
print(avg_age_rbdpaf)

std_age_rbdpaf=rbd_paf["Age"].std()
print(std_age_rbdpaf)

avg_age_msa=msa["Age"].mean()
print(avg_age_msa)

std_age_msa=msa["Age"].std()
print(std_age_msa)

# avg disease duration - not able to calculate without year symptoms began or date of diagnosis


# mean rnfl
avg_rnfl_controls=controls["Total Avgs"].mean()
print(avg_rnfl_controls)

avg_rnfl_rbdpaf=rbd_paf["Total Avgs"].mean()
print(avg_rnfl_rbdpaf)

avg_rnfl_msa=msa["Total Avgs"].mean()
print(avg_rnfl_msa)

#median rnfl
med_rnfl_controls=controls["Total Avgs"].median()
print(med_rnfl_controls)

med_rnfl_rbdpaf=rbd_paf["Total Avgs"].median()
print(med_rnfl_rbdpaf)

med_rnfl_msa=msa["Total Avgs"].median()
print(med_rnfl_msa)

#min rnfl
min_rnfl_controls=controls["Total Avgs"].min()
print(min_rnfl_controls)

min_rnfl_rbdpaf=rbd_paf["Total Avgs"].min()
print(min_rnfl_rbdpaf)

min_rnfl_msa=msa["Total Avgs"].min()
print(min_rnfl_msa)

# max rnfl
max_rnfl_controls=controls["Total Avgs"].max()
print(max_rnfl_controls)

max_rnfl_rbdpaf=rbd_paf["Total Avgs"].max()
print(max_rnfl_rbdpaf)

max_rnfl_msa=msa["Total Avgs"].max()
print(max_rnfl_msa)

# range rnfl
range_rnfl_controls=max_rnfl_controls-min_rnfl_controls
print(range_rnfl_controls)

range_rnfl_rbdpaf=max_rnfl_rbdpaf-min_rnfl_rbdpaf
print(range_rnfl_rbdpaf)

range_rnfl_msa=max_rnfl_msa-min_rnfl_msa
print(range_rnfl_msa)
