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
allDiagnosesGroupedMSA.name = "Overall"
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
def MOCA_23(df):
    if (df["MOCA_Score"] < 23):
        return 1
    elif (df["MOCA_Score"] >= 23):
        return 2
    else:
        return None
allDiagnoses["MOCA_23"] = allDiagnoses.apply(MOCA_23, axis = 1)
allDiagnosesGroupedMSA["MOCA_23"] = allDiagnosesGroupedMSA.apply(MOCA_23, axis = 1)
#%%
def UPSIT_18(df):
    if (df["UPSIT_Score"] < 18):
        return 1
    elif (df["UPSIT_Score"] >= 18):
        return 2
    else:
        return None
allDiagnoses["UPSIT_18"] = allDiagnoses.apply(UPSIT_18, axis = 1)
allDiagnosesGroupedMSA["UPSIT_18"] = allDiagnosesGroupedMSA.apply(UPSIT_18, axis = 1)
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

list_of_diagnoses = [allDiagnosesGroupedMSA, allMSA, PD, DLB, PAF, RBD, MSAC, MSAP]
#%%
def n_per_symptom(column):
    df = pd.DataFrame()
    for diagnosis in list_of_diagnoses:
        counts = diagnosis[column].value_counts()
        total = counts.sum()
        try:
            n = counts[1]
        except KeyError:
            n = 0

        percent = ((n / total) * 100).round(2)
        count_list = [n, percent]
        df[diagnosis.name] = count_list
    df = df.rename(index = {0 : "n", 1 : "%"})
    return df
#%%
def chi2_cont(column):
    df = pd.DataFrame()
    for diagnosis in list_of_diagnoses:
        counts = diagnosis[column].value_counts()
        total = counts.sum()
        try:
            yes = counts[1]
        except KeyError:
            yes = 0

        no = total - yes
        yes_no = [yes, no]
        df[diagnosis.name] = yes_no
    df = df.rename(index={0: "yes", 1: "no"})
    stat, pval, dof, expected = stats.chi2_contingency(observed=df)
    return df, pval
#%%
Neurological = {"N%" : {"MOCA < 23" : n_per_symptom("MOCA_23"), "UPSIT < 18" : n_per_symptom("UPSIT_18")}, "Chi-2" : {"MOCA < 23" : chi2_cont("MOCA_23"), "UPSIT < 18" : chi2_cont("UPSIT_18")}}
Parkinsonism = {"N%" : {"Bradykinesia" : n_per_symptom("Bradykinesia"), "Rigidity" : n_per_symptom("Rigidity"),
                  "Postural Instability" : n_per_symptom("Postural_Instability"), "Action Tremor" : n_per_symptom(
        "ActionTremor"), "Resting Tremor" : n_per_symptom("RestingTremor"), "Levo-Induced Dyskinesia/Dystonia" :
                         n_per_symptom("Dystonia")}, "Chi-2" : {"Bradykinesia" : chi2_cont("Bradykinesia"), "Rigidity" : chi2_cont("Rigidity"),
                  "Postural Instability" : chi2_cont("Postural_Instability"), "Action Tremor" : chi2_cont(
        "ActionTremor"), "Resting Tremor" : chi2_cont("RestingTremor"), "Levo-Induced Dyskinesia/Dystonia" :
                         chi2_cont("Dystonia")}}
Cerebellar = {"N%" : {"Gait Ataxia" : n_per_symptom("GaitAtaxia"), "Limb Ataxia" : n_per_symptom("LimbAtaxia"),
                   "Ataxic Dysarthria" : n_per_symptom("AtaxicDysarthria")}, "Chi-2" : {"Gait Ataxia" : chi2_cont("GaitAtaxia"), "Limb Ataxia" : chi2_cont("LimbAtaxia"),
                   "Ataxic Dysarthria" : chi2_cont("AtaxicDysarthria")}}
Autonomic = {"N%" : {"Urinary Incontinence" : n_per_symptom("UrinaryIncontinence"), "Incomplete Bladder Emptying" :
    n_per_symptom("IncompleteBladder"), "Constipation" : n_per_symptom("Constipation")}, "Chi-2" : {"Urinary Incontinence" : chi2_cont("UrinaryIncontinence"), "Incomplete Bladder Emptying" :
    chi2_cont("IncompleteBladder"), "Constipation" : chi2_cont("Constipation")}}
#%%
def n_per_OH():
    df30_15 = pd.DataFrame()
    df20_10 = pd.DataFrame()
    dfSH = pd.DataFrame()
    for diagnosis in list_of_diagnoses:
        counts = diagnosis["OH"].value_counts()
        total = 0
        for i in range(1, counts.size + 1):
            total += counts[i]

        n30_15 = counts[1] + counts[2]
        per30_15 = ((n30_15 / total) * 100).round(2)
        df30_15[diagnosis.name] = [n30_15, per30_15]

        n20_10 = counts[2]
        per20_10 = ((n20_10/ total) * 100).round(2)
        df20_10[diagnosis.name] = [n20_10, per20_10]

        nSH = counts[3]
        perSH = ((nSH / total) * 100).round(2)
        dfSH[diagnosis.name] = [nSH, perSH]

    df30_15 = df30_15.rename(index={0: "n", 1: "%"})
    df20_10 = df20_10.rename(index={0: "n", 1: "%"})
    dfSH = dfSH.rename(index={0: "n", 1: "%"})
    OH_nPER = {"30/15" : df30_15, "20/10" : df20_10, "SH": dfSH}
    return OH_nPER

OH_nPER = n_per_OH()
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

        #tukey_to_latex(dict, variable)
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
#%%
UMSARS_dict = test_dict(["UMSARS_1_Score", "UMSARS_2_Score"])
#%%
GDS_dict = test_dict(["UMSARS_4_DisabilityScale"])

