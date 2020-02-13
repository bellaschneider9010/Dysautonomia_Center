#%%
# IMPORTANT INFORMATION:
# Multiple System Atrophy (MSA)  is rare disease that is similar to Parkinson's, Disease (PD)  but presents differently in certain ways. For example, the disease progresses much faster than PD, and patients with MSA often don't respond to medications to treat motor symptoms (like Levadopa-Carbidopa). In this code, I am comparing differences in data between patients at study enrollment with MSA-P and MSA-C, which are the two sub-types (Parkinson's and Cerebellar, respectively).
# This source code was used to fill out slides for a presentation. To see the relevant slides, please see MSA-C vs MSA-P (Sample).pptx
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
from scipy import stats
#%%
data=pd.read_csv("MSA-P vs MSA-C (Sample).csv")
data=data.dropna(subset=["Local ID"])   #removes all rows that don't have an ID
#%%
data["cSBP"] = abs(data["upSBP"] - data["supSBP"])
data["cDBP"] = abs(data["upDBP"] - data["supDBP"])
data["cHR"]=abs(data["upHR"] - data["supHR"])
data["cNE"]=abs(data["upNE"] - data["supNE"])
#add new columns that calculate the magnitude of change between Upright Systolic Blood Pressure (upSBP) and Supine Systolic Blood Pressure (supSBP). Same was done for Upright Diastolic Blood Pressure (upDBP) and Supine Diastolic Blood Pressure (supDBP). Measures taken from the Orthostatic Hypotension Tilt Test
data.name="data"
#%%
def conditions(df):
    if (df["cSBP"]>=30) and (df["cDBP"]>=15):
        return 1
    elif (20<=df["cSBP"] <30) and (10<=df["cDBP"] <15):
        return 2
    elif (df["supSBP"]>140) and (df["supDBP"]>90):
        return 3
    else:
        return None
data["OH"]=data.apply(conditions, axis=1)
# Creates a column that shows if they have Orthostatic Hypotension (1 or 2) or Supine Hypertension (3). Orthostatic Hypotension is defined as having a 20/10 decrease or more in blood pressure upon standing up. We also note if they had a 30/15 decrease or more in blood pressure. Supine Hypertension is defined as having a supine blood pressure over 140/90.
#%%
def MOCA23(df):
    if df["MOCA"] <23:
        return 1
    else:
        return None
data["MOCA23"]=data.apply(MOCA23,axis=1)
#"Mild Cognitive Impairment" (MCI) is defined as having a Montreal Cognitive Assessment (MoCA) score of under 23 (scored out of 30).
#%%
def UPSIT18(df):
    if df["UPSIT"]<18:
        return 1
    else:
        return None
data["UPSIT18"]=data.apply(UPSIT18, axis=1)
# The UPSIT is a smell test. If a patient has a score less than 18 (scored out of 40), then the patients have anosmia.
#%%
def UMSARS10(df):
    if df["10. Urinary function"] >=2:
        return 1
    else:
        return None
data["UMSARS10"]=data.apply(UMSARS10, axis=1)
# Refers to Question 10 of the UMSARS (neurological symptom rating scale). A score of two or more means that they require treatment for their urinary incontinence.
#%%
def UMSARS12(df):
    if df["12. Bowel Function"] >=1:
        return 1
    else:
        return None
# Refers to Question 12 of the USMARS. A score of 1 or higher means they suffer from constipation, possibly requiring lasxitives.
data["UMSARS12"]=data.apply(UMSARS12, axis=1)
#
#%%
#Creates three dataframes; Overall ("data"), patients with MSA-P ("msap"), and MSA-C ("msac")
msap=data.loc[lambda data: data["Diagnosis"]=="P"]
msap.name="msap"
msac=data.loc[lambda data: data["Diagnosis"]=="C"]
msac.name="msac"
#%%
order = [data, msap, msac]
#%%
def OH(type):
    counts = type["OH"].value_counts()
    print(counts)
    print(type["OH"].count())
    print("30/15:", counts[1] + counts[2], "20/10:", counts[2])
    print("%30/15:", ((counts[1] + counts[2]) / type["OH"].count()) * 100, "%20/10:",
          (counts[2] / type["OH"].count()) * 100)
    print("Supine HT:", counts[3], (counts[3] / type["OH"].count()) * 100)
# Calculates how many people have OH and/or Supine HT.
#%%
def percent(sub,total):
    return (sub.shape[0]/total.shape[0])*100
#%%
def  bygender(gender):
    print(gender)
    for type in [msap,msac]:
        print(type.name)
        count=type.loc[type["Gender"]==gender]
        print(count.shape[0], percent(count,type))
# Note that this refers to biological sex, not gender identity.
#%%
def make_array(column):
    p=msap[column].to_numpy(dtype=float)
    c=msac[column].to_numpy(dtype=float)
    return p,c
#%%
# calculates the mean and the standard deviation for a column for each dataframe
def mean_std(variable):
    print(variable)
    dforder = [data, msap, msac]
    for i in dforder:
        print(i.name)
        print("n=", i[variable].count())
        print(i[variable].dropna().shape[0])
        print("mean:", i[variable].mean(), "std:", i[variable].std())
#%%
# calculates percentage of incidence of a symptom in each dataframe
def SF_count(column):
    print(column)
    for df in order:
        counts = df[column].value_counts()
        print(df[column].count())
        print(df.name, (counts[1], counts[1]/(counts[1]+counts[2])*100))
#%%
# calculates percentage of each score on the UMSARS IV: Global Disability Scale (scored out of 5)
def disability_scale():
    for type in order:
        print(type.name)
        print(type["Disability Scale"].count())
        counts = type["Disability Scale"].value_counts()
        for i in range(5):
            print(i, counts[i])
            print("%n", (counts[i] / type["Disability Scale"].count()) * 100)
#%%
for df in order:
    df.shape[0]
#%% Here is where I start analyzing the data. It is quite messy because I wrote and ran the code as I filled out the slides; section-by-section.
for type in [msap,msac]:
    percent(type,data)
#%%
bygender("Female")
bygender("Male")
#%%
genderchi = np.array([[77, 95], [78, 99]])
print(genderchi)
stats.chi2_contingency(genderchi)
#%%
for age in ["Age Baseline", "Age at Onset", "Duration of Illness"]:
   mean_std(age)
   (p, c) = make_array(age)
   stats.ttest_ind(p, c, nan_policy="omit")
#%%
for BP in ["supSBP","upSBP", "cSBP","supDBP", "upDBP", "cDBP", "supHR", "upHR", "cHR", "supNE", "upNE", "cNE"]:
    mean_std(BP)
    (p, c) = make_array(BP)
    stats.ttest_ind(p, c, nan_policy="omit")
#%%
for df in order:
    print(df.name)
    OH(df)
#%%
supHT=np.array([[12, 19], [58, 56]])
stats.chi2_contingency(supHT)
#%%
for column in ["Urinary Incontinence", "Incomplete Bladder", "Constipation"]:
    SF_count(column)
#%%
ui = np.array([[103, 78], [22, 42]])
stats.chi2_contingency(ui)
ibe= np.array([[89, 80], [39, 43]])
stats.chi2_contingency(ibe)
con=np.array([[120, 99], [14, 30]])
stats.chi2_contingency(con)
#%%
mean_std("Valsalva Ratio")
(p, c) = make_array("Valsalva Ratio")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
mean_std("E:I Ratio")
(p, c) = make_array("E:I Ratio")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
mean_std("MOCA")
(p, c) = make_array("MOCA")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
def moca23(type):
    counts = type["MOCA23"].value_counts()
    print("MOCA ALL:", type["MOCA"].count())
    print("MoCA <23: ", counts[1])
    print("%n", (counts[1] / type["MOCA"].count()) * 100)
for df in order:
    print(df.name)
    moca23(df)
#%%
moca23chi=np.array([[27,27], [133, 141]])
stats.chi2_contingency(moca23chi)
#%%
mean_std("UPSIT")
(p, c) = make_array("UPSIT")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
def upsit18(type):
    print(type["UPSIT"].count())
    counts = type["UPSIT18"].value_counts()
    print("UPSIT <18: ", counts[1])
    print("%n", (counts[1] / type["UPSIT"].count()) * 100)
for df in order:
    print(df.name)
    upsit18(df)
#%%
upsit18chi=np.array([[16,11],[91,101]])
stats.chi2_contingency(upsit18chi)
#%%
mean_std("UMSARS I")
(p, c) = make_array("UMSARS I")
stats.ttest_ind(p, c, nan_policy="omit")
mean_std("UMSARS II")
(p, c) = make_array("UMSARS II")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
def umsars10(type):
    counts = type["UMSARS10"].value_counts()
    print("All 10:", type["10. Urinary function"].count())
    print("UMSARS <=2: ", counts[1])
    print("%n", (counts[1] / type["10. Urinary function"].count()) * 100)
for df in order:
    print(df.name)
    umsars10(df)
#%%
umsars10chi=np.array([[124, 95], [38, 72]])
stats.chi2_contingency(umsars10chi)
#%%
def umsars12(type):
    counts = type["UMSARS12"].value_counts()
    print(counts)
    print("All 12:", type["12. Bowel Function"].count())
    print("UMSARS <=1: ", counts[1])
    print("%n", (counts[1] / type["12. Bowel Function"].count()) * 100)
for df in order:
    print(df.name)
    umsars12(df)
#%%
umsars12chi=np.array([[144, 130], [18, 37]])
stats.chi2_contingency(umsars12chi)
#%%
mean_std("Disability Scale")
(p, c) = make_array("Disability Scale")
stats.ttest_ind(p, c, nan_policy="omit")
#%%
for i in range(5):
    disability_scale()
#%%
disabilitychi=np.array([[12,23],[55,65],[39,40],[43,29],[3,3]])
stats.chi2_contingency(disabilitychi)
#%%
for i in ["Motor Score", "Non-Motor Score", "Mood Score"]:
    mean_std(i)
    (p, c) = make_array(i)
    stats.ttest_ind(p, c, nan_policy="omit")
#%%
for column in ["Bradykinesia","Rigidity","Postural Instability", "Postural/action tremor", "Resting tremor", "Levodopa-induced dyskinesia/dystonia"]:
    SF_count(column)
#%%
bk=np.array([[121, 83], [4, 34]])
stats.chi2_contingency(bk)
#%%
rigidity=np.array([[117, 64], [6, 54]])
stats.chi2_contingency(rigidity)
#%%
instability=np.array([[12, 3], [145, 164]])
stats.chi2_contingency(instability)
#%%
action=np.array([[76, 64], [48, 58]])
stats.chi2_contingency(action)
#%%
resting = np.array([[43, 18], [82, 103]])
stats.chi2_contingency(resting)
#%%
levodys=np.array([[17, 1], [95, 86]])
stats.chi2_contingency(levodys)
#%%
for column in ["Gait Ataxia", "Limb Ataxia", "Ataxic Dysarthria"]:
    SF_count(column)
#%%
gait=np.array([[38, 188], [78, 4]])
stats.chi2_contingency(gait)
#%%
limb=np.array([[36, 113], [86, 7]])
stats.chi2_contingency(limb)
#%%
ataxic = np.array([[33, 108], [87, 8]])
stats.chi2_contingency(ataxic)
#%%
for column in ["Babinski Sign", "Hyperreflexia"]:
    SF_count(column)
#%%
babinski=np.array([[45, 41], [72, 65]])
stats.chi2_contingency(babinski)
#%%
hyperreflexia=np.array([[60, 72], [55, 38]])
stats.chi2_contingency(hyperreflexia)
#%%
SF_count("M_RBD")
#%%
rbd=np.array([[92, 101], [17, 16]])
stats.chi2_contingency(rbd)
