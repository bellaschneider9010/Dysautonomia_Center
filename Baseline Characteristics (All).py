#%%
# This code compares blood pressure measures from the Tilt Test between patients with different diagnoses. 
#Results of the one-way ANOVA, Tukey-Kramer, and T-test have been put in dictionaries.
#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
from scipy import stats
from pingouin import pairwise_tukey
#%%
allDiagnoses=pd.read_csv("Baseline Characteristics (All).csv")
allDiagnoses=allDiagnoses.dropna(subset=["LocalID"])
#%%
allDiagnoses["cSBP"] = abs(allDiagnoses["upSBP"] - allDiagnoses["supSBP"])
allDiagnoses["cDBP"] = abs(allDiagnoses["upDBP"] - allDiagnoses["supDBP"])
allDiagnoses["cHR"]=abs(allDiagnoses["upHR"] - allDiagnoses["supHR"])
allDiagnoses["cNE"]=abs(allDiagnoses["upNE"] - allDiagnoses["supNE"])
allDiagnoses.name="allDiagnoses"

#%%
allDiagnosesGroupedMSA = allDiagnoses.copy()
allDiagnosesGroupedMSA.ClinicalDiagnosis.replace(["MSAC", "MSAP"], ["MSA", "MSA"],
                                                                          inplace = True)
allDiagnosesGroupedMSA.name = "allDiagnosesGroupedMSA"
#%%
def conditions(df):
    if (df["cSBP"]>=30) or (df["cDBP"]>=15):
        return 1
    elif (20<=df["cSBP"] <30) or (10<=df["cDBP"] <15):
        return 2
    elif (df["supSBP"]>140) or (df["supDBP"]>90):
        return 3
    elif (df["cSBP"] >= 0.0) or (df["cDBP"] >= 0.0):
        return 4
    else:
        return None
allDiagnoses["OH"]=allDiagnoses.apply(conditions, axis=1)
allDiagnosesGroupedMSA["OH"] = allDiagnosesGroupedMSA.apply(conditions, axis=1)
#%%
DLB = allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "DLB"]
DLB.name = "DLB"
MSAP = allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "MSAP"]
MSAP.name = "MSAP"
MSAC = allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "MSAC"]
MSAC.name = "MSAC"
allMSA =  allDiagnoses.loc[(allDiagnoses["ClinicalDiagnosis"] == "MSAC") | (allDiagnoses["ClinicalDiagnosis"] ==
                                                                            "MSAP")]
allMSA.name = "allMSA"
PD= allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "PD"]
PD.name = "PD"
PAF = allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "PAF"]
PAF.name = "PAF"
RBD = allDiagnoses.loc[lambda allDiagnoses: allDiagnoses["ClinicalDiagnosis"] == "RBD"]
RBD.name = "RBD"
#%%
def OH(dataframe):
    print(dataframe.name)
    counts = dataframe["OH"].value_counts()
    total = counts[1] + counts[2] + counts[3] + counts[4]
    n30_15 = counts[1] + counts[2]
    per30_15 = (((counts[1] + counts[2]) / dataframe["OH"].count()) *
                                                        100).round(2)
    n20_10 = counts[2]
    per20_10 = ((counts[2] / dataframe["OH"].count()) * 100).round(2)
    no_OH = total - n30_15
    nSupHT = counts[3]
    perSupHT = ((counts[3] / dataframe["OH"].count()) * 100).round(2)
    no_SupHT = total - nSupHT
    print("n30/15:", n30_15, "%30/15:", per30_15)
    print("n20/10:", n20_10, "%20/10:", per20_10)
    print("Supine HT:", nSupHT, perSupHT)
    OH_array = [n30_15, no_OH]
    SH_array = [nSupHT, no_SupHT]
    return total, n30_15, per30_15, n20_10, per20_10, nSupHT, perSupHT, OH_array, SH_array
#%%
for dfs in [allDiagnosesGroupedMSA, allMSA, PD, DLB, PAF, RBD, MSAC, MSAP]:
    OH(dfs)
#%%
OH_array = OH(allDiagnosesGroupedMSA)[7]
OH_df = pd.DataFrame({'Overall': OH_array})
for dfs in [allMSA, PD, DLB, PAF, RBD]:
    df2 = pd.DataFrame({dfs.name : OH(dfs)[7]})
    OH_df = OH_df.join(df2)
OH_df = OH_df.rename(index = {0 : "has OH", 1 : "no OH"})
stats.chi2_contingency(observed=OH_df)
#%%
SH_array = OH(allDiagnosesGroupedMSA)[8]
SH_df = pd.DataFrame({'Overall': SH_array})
for dfs in [allMSA, PD, DLB, PAF, RBD]:
    df2 = pd.DataFrame({dfs.name : OH(dfs)[8]})
    SH_df = SH_df.join(df2)
SH_df = SH_df.rename(index = {0 : "has SH", 1 : "no SH"})
stats.chi2_contingency(observed=SH_df)
#%%
OH_array_MSAs = OH(MSAC)[7]
OH_df_MSA = pd.DataFrame({'MSA-C': OH_array})
df_MSAP = pd.DataFrame({MSAP.name : OH(MSAP)[7]})
OH_df_MSA = OH_df_MSA.join(df_MSAP)
OH_df_MSA = OH_df_MSA.rename(index = {0 : "has OH", 1 : "no OH"})
stats.chi2_contingency(observed=OH_df_MSA)
#%%
SH_array_MSAs = OH(MSAC)[8]
SH_df_MSA = pd.DataFrame({'MSA-C': SH_array_MSAs})
df_MSAP_sh = pd.DataFrame({MSAP.name : OH(MSAP)[8]})
SH_df_MSA = SH_df_MSA.join(df_MSAP)
SH_df_MSA = SH_df_MSA.rename(index = {0 : "has SH", 1 : "no SH"})
stats.chi2_contingency(observed=SH_df_MSA)
#%%
count = allDiagnoses["supSBP"].count()
#%%
def mean_sd_df(variable):
    dfDiagnoses = [allMSA, PD, DLB, PAF, RBD,MSAC, MSAP]
    Diagnoses = ["Overall", "all MSA", "PD", "DLB",  "PAF", "RBD", "MSA-C", "MSA-P"]
    count = allDiagnoses[variable].count()
    array = np.array([count, allDiagnoses[variable].mean(), allDiagnoses[
        variable].std()])
    for diagnosis in dfDiagnoses:
        count = diagnosis[variable].count()
        row = np.array([count, diagnosis[variable].mean(), diagnosis[variable].std()])
        array = np.vstack((array, row))
    mean_sd = pd.DataFrame(data = array, columns = ["n","Mean", "SD"], index = Diagnoses).round(
        2).applymap("{0:.2f}".format)
    return mean_sd
#%%
def anova_tukey(dataframe, column):
    df = dataframe[["ClinicalDiagnosis", column]]
    df_pivot = df.pivot(columns = "ClinicalDiagnosis", values = column)
    data = [df_pivot[diagnosis].dropna().values for diagnosis in df_pivot]
    f_val, p_val = stats.f_oneway(*data)
    anova_results = f_val, p_val
    tukey = pairwise_tukey(data=df, dv=column, between="ClinicalDiagnosis")
    return anova_results, tukey
#%%
def ttest_MSAs(variable):
    MSAC_clean = MSAC[variable].dropna()
    MSAP_clean = MSAP[variable].dropna()
    MSAC_array = np.asarray(MSAC_clean).transpose()
    MSAP_array = np.asarray(MSAP_clean).transpose()
    stat, pval = stats.ttest_ind(MSAC_array, MSAP_array)
    return stat, pval
#%%
# Function to export a dataframe to .tex/.pdf
def tukey_to_latex(dict, variable):
    copy = dict["tukey"].copy()
    copy[variable] = copy[variable].drop(columns=["hedges", "se", "T", "tail"])
    for eachrow in copy[variable]["p-tukey"]:
        copy[variable]["Significance"] = copy[variable]["p-tukey"].map(lambda eachrow: "n.s" if eachrow > 0.05 else "*" if 0.001 < eachrow <= 0.05 else "**")

    table_name = variable + ".tex"
    beginningtex = """\\documentclass{report}
    \\usepackage{booktabs}
    \\usepackage{caption}
    \\begin{document}
    \\begin{center}
    \\begin{table}[ht]
    \\caption*{mycaption}
    \\noindent\makebox[\\textwidth]{
    """
    endtex = """}
    \\end{table}
    \\end{center}
    \\end{document}"""

    f = open(table_name, 'w')
    f.write(beginningtex)
    f.write(copy[variable].to_latex(index=False, float_format="{:0.4f}".format, column_format="llccccc"))
    f.write(endtex)
    f.close()
#%%
# Creates a dictionary from dictionaries of each test performed for each variable. Exports Tukey-Kramer test ("tukey")
# to .tex/.pdf
def test_dict(array):
    dict = {"mean_sd": {}, "tukey": {}, "ttest_MSAs": {}, "anova_results" : {}}
    for variable in array:
        dict["mean_sd"][variable] = mean_sd_df(variable)
        dict["tukey"][variable] = anova_tukey(allDiagnosesGroupedMSA, variable)[1]
        dict["ttest_MSAs"][variable] = ttest_MSAs(variable)
        dict["anova_results"][variable] = anova_tukey(allDiagnosesGroupedMSA, variable)[0]

        tukey_to_latex(dict, variable)
    return dict
#%%
Neuro_dict = test_dict(["MOCA_Score", "UPSIT_Score"])
#%%
TiltTest_dict = test_dict(["supSBP", "upSBP", "cSBP", "supDBP", "upDBP", "cDBP", "supHR", "upHR", "cHR", "supNE",
                           "upNE", "cNE"])
#%%
Autonomic_dict = test_dict(["ValsalvaRatio", "EIRatio"])
#%%
QOL_dict = test_dict(["QOL_Motor_Score", "QOL_NonMotor_Score", "QOL_Mood_Score", "QOL_Total_Score"])
