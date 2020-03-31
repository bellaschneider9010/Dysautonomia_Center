#%%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import math
import numpy as np
from scipy import stats
from pingouin import pairwise_tukey
from collections import OrderedDict
from itertools import islice
import datetime
import os
#%%
todayDate = datetime.datetime.today().strftime('%m-%d-%Y')
folderName = todayDate # Put the name of the enclosing folder in between quotations. If the name of the folder is today's date, then use todayDate, no quotations.
p = os.path.abspath(folderName)
cwd = os.getcwd()
if cwd != p:
  os.chdir(folderName)
os.chdir("03-26-2020")
#%%
rawdata = pd.read_csv("Baseline Chars Feb 2020.csv", parse_dates= ["consentstatusdt", "docode"]).dropna(subset = ["localid"])
# Put the CSV file name  between the quotations
#%%
def fix_date(x):
    if x.year > 1989:
        year = int(x.year - 100)
        return datetime.date(year, x.month, x.day)
    else:
        try:
            year = int(x.year)
            return datetime.date(year, x.month, x.day)
        except ValueError:
            return np.nan
rawdata["docode"] = rawdata["docode"].apply(fix_date)
rawdata["docode"] = pd.to_datetime(rawdata["docode"])
#%%
baseline = rawdata.loc[lambda rawdata: rawdata["redcap_event_name"] == "1_entry_visit_arm_1"]
baseline = baseline.drop("redcap_event_name", axis = 1).reset_index(drop = True)
#%%
baseline["ageAtEnrollment"] = baseline['consentstatusdt'].sub(baseline['docode'], axis=0) / np.timedelta64(1, 'Y')
baseline["yearOfBirth"] = baseline["docode"].apply(lambda x : x.year)
#%%
baseline = baseline.rename(columns = {"afttilheadup3sys" : "upSBP", "afttiltsupine0sys" : "supSBP", "afttiltheadup3dias" : "upDBP", "afttiltsupine0dias" : "supDBP", "afttiltheadup3hr" : "upHR", "afttiltsupine0hr": "supHR", "catuprightne" : "upNE", "catssupinene" : "supNE"})
#%%
for column in ["diag_entry", "gender", "hxpisymptoms", "upSBP", "supSBP", "upDBP", "supDBP", "upHR", "supHR", "upNE", "supNE", "aftvalsalvaratio", "aftrrdeeppacedeiratio", "motor_score", "non_motor", "mood_score", "total_qol_score", "umsars1scocalc", "umsars2scocalc", "umsarsadl10", "umsarsadl12", "umsarsdis1", "moca_sco", "upsit"]:
    for index in baseline.index:
            try:
                baseline.at[index, column] = float(baseline.at[index, column])
            except ValueError:
                baseline.at[index, column] = np.nan
    baseline[column].astype(float).dtypes

del(column, index)
#%%
baseline["ageAtOnset"] = baseline["hxpisymptoms"] - baseline["yearOfBirth"]
for value in baseline["ageAtOnset"]:
    if value < 30:
        baseline["ageAtOnset"].replace(value, np.nan, inplace = True)
#%%
def cleanColumn(column1, column2):
    column1Clean = column1 + "clean"
    column2Clean = column2 + "clean"
    for index in baseline.index:
        if (math.isnan(baseline.at[index, column1]) == False) and (math.isnan(baseline.at[index, column2]) == False):
            baseline.at[index, column1Clean] = baseline.at[index, column1]
            baseline.at[index, column2Clean] = baseline.at[index, column2]
        else:
            baseline.at[index, column1Clean] = np.nan
            baseline.at[index, column2Clean] = np.nan
#%%
cleanColumn("ageAtOnset", "ageAtEnrollment")
baseline["durationOfSymptoms"] = baseline["ageAtEnrollment"] - baseline["ageAtOnset"]
cleanColumn("upSBP", "supSBP")
baseline["cSBP"] = baseline["upSBPclean"] - baseline["supSBPclean"]
cleanColumn("upDBP", "supDBP")
baseline["cDBP"] = baseline["upDBPclean"] - baseline["supDBPclean"]
cleanColumn("upHR", "supHR")
baseline["cHR"] = baseline["upHRclean"] - baseline["supHRclean"]
cleanColumn("upNE", "supNE")
baseline["cNE"] = baseline["upNEclean"] - baseline["supNEclean"]
baseline["cHR/cSBP"] = abs(baseline["cHR"] / baseline["cSBP"])
#%%
def HRdivSBP(df):
    if df["cHR/cSBP"] < 0.5:
        return 1
    elif (df["cHR/cSBP"] >= 0.0):
        return 2
    else:
        return None
baseline["nOHratio"]=baseline.apply(HRdivSBP, axis=1)
#%%
def OH_type(df):
    if (df["cSBP"] <= -30) or (df["cDBP"] <= -15):
        return 1
    elif (-30 < df["cSBP"] <= -20) or (-15 < df["cDBP"] <= -10):
        return 2
    if (np.isnan(df["cSBP"]) == False) or (np.isnan(df["cDBP"]) == False):
        return 3
    else:
        return None
baseline["typeOH"] = baseline.apply(OH_type, axis=1)
#%%
def hasOH(df):
    if (df["typeOH"] == 1) or (df["typeOH"] == 2):
        return 1
    elif df["typeOH"] == 3:
        return 2
    else:
        return None
baseline["hasOH"] = baseline.apply(hasOH, axis = 1)
#%%
def hasSH(df):
    if (df["supSBP"] > 140) or (df["supDBP"] > 90):
        return 1
    elif (np.isnan(df["supSBP"]) == False) or (np.isnan(df["supDBP"]) == False):
        return 2
    else:
        return None
baseline["hasSH"] = baseline.apply(hasSH, axis=1)
#%%
def UMSARSurinaryFunc(df):
    if (df["umsarsadl10"] >= 2):
        return 1
    elif (df["umsarsadl10"] < 2):
        return 2
    else:
        return None
baseline["UMSARSurinaryFunc"] = baseline.apply(UMSARSurinaryFunc, axis = 1)
#%%
def UMSARSbowelFunc(df):
    if (df["umsarsadl12"] >= 1):
        return 1
    elif (df["umsarsadl12"] < 1):
        return 2
    else:
        return None
baseline["UMSARSbowelFunc"] = baseline.apply(UMSARSbowelFunc, axis = 1)
#%%
def MOCA_23(df):
    if (df["moca_sco"] < 23):
        return 1
    elif (df["moca_sco"] >= 23):
        return 2
    else:
        return None
baseline["MOCA_23"] = baseline.apply(MOCA_23, axis = 1)
#%%
def UPSIT_18(df):
    if (df["upsit"] < 18):
        return 1
    elif (df["upsit"] >= 18):
        return 2
    else:
        return None
baseline["UPSIT_18"] = baseline.apply(UPSIT_18, axis = 1)
#%%
baseline = baseline.rename(columns = {"diag_entry" : "diagnosis"})
#%%
baseline.diagnosis.replace([1, 2], "PAF", inplace = True)
baseline.diagnosis.replace(3, "MSAC", inplace = True)
baseline.diagnosis.replace(4, "MSAP", inplace = True)
baseline.diagnosis.replace([5, 6], "PD", inplace = True)
baseline.diagnosis.replace(7, "DLB", inplace = True)
baseline.diagnosis.replace(8, "RBD", inplace = True)
baseline.diagnosis.replace(9, np.nan, inplace = True)
baseline = baseline.dropna(subset = ["diagnosis"])
#%%
diagnoses = {}

diagnoses["Overall"] = baseline.copy()
diagnoses["Overall"].replace(["MSAC", "MSAP"], "MSA", inplace = True)

diagnoses["MSA"] = diagnoses["Overall"].loc[diagnoses["Overall"]["diagnosis"] == "MSA"]

diagnoses["PD"] = baseline.loc[baseline["diagnosis"] == "PD"]

diagnoses["DLB"] = baseline.loc[baseline["diagnosis"] == "DLB"]

diagnoses["PAF"] = baseline.loc[baseline["diagnosis"] == "PAF"]

diagnoses["RBD"] = baseline.loc[baseline["diagnosis"] == "RBD"]

diagnoses["MSAC"] = baseline.loc[baseline["diagnosis"] == "MSAC"]

diagnoses["MSAP"] = baseline.loc[baseline["diagnosis"] == "MSAP"]

for name, diagnosis in diagnoses.items():
    diagnosis.name = name
del (name, diagnosis)
#%%
baseline.loc[:, "prog_disease" : "speakact_dreams"].replace([2, 3], 2, inplace = True)
#%%
NperDiagnosis = pd.DataFrame()
for name, diagnosis in diagnoses.items():
    counts = diagnosis.shape[0]
    NperDiagnosis[diagnosis.name] = [counts]
NperDiagnosis = NperDiagnosis.rename(index = {0 : "n"})
del (name, diagnosis, counts)
#%%
def sexChiCounts(column):
    nper = pd.DataFrame()
    df = pd.DataFrame()
    for name, diagnosis in islice(diagnoses.items(), len(diagnoses)-2):
        counts = diagnosis[column].value_counts()
        total = counts.sum()
        male = counts[1]
        female =  counts[2]
        nper[diagnosis.name] = [male, female]
    stat, pval, dof, expected = stats.chi2_contingency(observed=nper)
    for name, diagnosis in diagnoses.items():
        counts = diagnosis[column].value_counts()
        total = counts.sum()
        male = counts[1]
        female = counts[2]
        perMale = ((male / total) * 100).round(2)
        perFemale = ((female / total) * 100).round(2)
        df[diagnosis.name] = [male, perMale, female, perFemale]
    df = df.rename(index = {0: "n (Male)", 1: "% (Male)", 2: "n (Female)", 3: "% (Female)"})
    dict = {"N/%" : df, "Chi-2" : pval}
    return dict
sexChiCounts = sexChiCounts("gender") # Does not include MSA-C and MSA-P in Chi-2 analysis
#%%
def chiCounts(column):
    nper = pd.DataFrame()
    df = pd.DataFrame()
    for name, diagnosis in islice(diagnoses.items(), len(diagnoses)-2):
        counts = diagnosis[column].value_counts()
        total = counts.sum()
        try:
            n = counts[1]
        except KeyError:
            n = 0
        percent = ((n / total) * 100).round(2)
        no = total - n
        nper[diagnosis.name] = [n, no]
        df[diagnosis.name] = [n, percent, no]
    stat, pval, dof, expected = stats.chi2_contingency(observed = nper)
    df = df.rename(index = {0: "yes", 1: "%", 2: "no"})
    dict = {"N/%" : df, "Chi-2" : pval}
    return dict
#%%
# Disregard the Chi-2 for OH-30/15 and OH-20/10, for the groups are NOT independent.

OH_SH_nPER = {"OH-30/15" : chiCounts("typeOH"), "OH-20/10" : chiCounts("hasOH"), "Supine Hypertension" : chiCounts("hasSH")}

Autonomic = {"Urinary Incontinence" : chiCounts("uri_incon"), "Incomplete Bladder Emptying" : chiCounts("incompbladderemp"), "Constipation": chiCounts("consip_checklist")}

Parkinsonism = {"Bradykinesia" : chiCounts("bradykicheck"), "Rigidity" : chiCounts("rigid"), "Action Tremor" : chiCounts("posturaltrem"), "Resting Tremor": chiCounts("rest_trem"), "Postural Instability" : "post_instab", "Levo-Induced dyskinesia/dystonia" : chiCounts("levo_induced_dyski")}

Cerebellar = {"Gait Ataxia" : chiCounts("gaitatax"), "Limb Ataxia" : chiCounts("limbatax"), "Ataxic Dysarthria" : chiCounts("ataxdysarth")}

nOH_Ratio = chiCounts("nOHratio")

hasRBD = chiCounts("screen_rbd")

#%%
def meanSD(variable):
    indices = []
    array = np.empty((0, 3), float)
    for name, diagnosis in diagnoses.items():
        count = diagnosis[variable].count()
        row = np.array([count, diagnosis[variable].mean(), diagnosis[variable].std()])
        array = np.vstack((array, row))
        indices.append(diagnosis.name)

    mean_sd = pd.DataFrame(data = array, columns = ["n", "Mean", "SD"], index = indices).round(
        2).applymap("{0:.2f}".format)
    return mean_sd
#%%
def ANOVA_Tukey(variable):
    new = diagnoses["Overall"][["diagnosis", variable]]
    new = new.astype({variable: 'float64'})
    df_pivot = new.pivot(columns = "diagnosis", values = variable)
    data = [df_pivot[diagnosis].dropna().values for diagnosis in df_pivot]
    f_val, p_val = stats.f_oneway(*data)
    anova_results = [f_val, p_val]
    tukey = pairwise_tukey(data = new, dv = variable, between = "diagnosis")
    return anova_results, tukey
#%%
def TtestMSA(variable):
    MSAC_clean = diagnoses["MSAC"][variable].dropna()
    MSAP_clean = diagnoses["MSAP"][variable].dropna()
    MSAC_array = np.asarray(MSAC_clean).transpose()
    MSAP_array = np.asarray(MSAP_clean).transpose()
    stat, pval = stats.ttest_ind(MSAC_array, MSAP_array)
    return stat, pval
#%%
def tukeyToLatex(dictionary, variable):
    copy = dictionary.copy()
    copy = copy.drop(columns=["hedges", "se", "T", "tail"])
    for eachrow in copy["p-tukey"]:
        copy["Significance"] = copy["p-tukey"].map(lambda eachrow: "n.s" if eachrow > 0.05 else "*" if 0.001 < eachrow <= 0.05 else "**")

    table_name = variable + ".tex"
    caption = variable.replace("_", " ")
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
    f.write(beginningtex.replace("mycaption", "Tukey-Kramer analysis of " + caption))
    f.write(copy.to_latex(index=False, float_format="{:0.4f}".format, column_format="llccccc"))
    f.write(endtex)
    f.close()
#%%
def testDict(variable):
    dict = {"Mean + SD": {}, "ANOVA": {}, "Tukey-Kramer": {}, "T-test (MSA-P v. MSA-C)": {}}
    dict["Mean + SD"] = meanSD(variable)
    dict["ANOVA"] = ANOVA_Tukey(variable)[0]
    dict["Tukey-Kramer"] = ANOVA_Tukey(variable)[1]
    dict["T-test (MSA-P v. MSA-C)"] = TtestMSA(variable)

    tukeyToLatex(dict["Tukey-Kramer"], variable)

    return dict

#%%
TiltTest = {"Systolic" : {"Supine" : testDict("supSBP"), "Upright" : testDict("upSBP"), "Change" : testDict("cSBP")}, "Diastolic" : {"Supine" : testDict("supDBP"), "Upright" : testDict("upDBP"), "Change" : testDict("cDBP")}, "Heart Rate" : {"Supine" : testDict("supHR"), "Upright" : testDict("upHR"), "Change" : testDict("cHR")}, "Norepinephrine" : {"Supine" : testDict("supNE"), "Upright" : testDict("upNE"), "Change" : testDict("cNE")}}

RatiosTiltTest = {"Valsalva Ratio" : testDict("aftvalsalvaratio"), "EI Ratio" : testDict("aftrrdeeppacedeiratio")}

QOL = {"Motor Score" : testDict("motor_score"), "Non-Motor Score" : testDict("non_motor"), "Mood Score" : testDict("mood_score"), "Total Score" : testDict("total_qol_score")}

UMSARS = {"UMSARS-I Score" : testDict("umsars1scocalc"), "UMSARS-II Score": testDict("umsars2scocalc"), "Global Disability Scale" : testDict("umsarsdis1"), "UMSARS-I Q10 < 2" : chiCounts("UMSARSurinaryFunc"), "UMSARS-I Q12 < 1" : chiCounts("UMSARSbowelFunc")}

Neurological = {"MOCA" : testDict("moca_sco"), "UPSIT": testDict("upsit"), "MOCA < 23" : chiCounts("MOCA_23"), "UPSIT < 18": chiCounts("UPSIT_18")}

SymptomsTime = {"Age at Enrollment" : testDict("ageAtEnrollment"), "Age at Symptom Onset" : testDict("ageAtOnset"), "Duration of Symptoms" : testDict("durationOfSymptoms")}
#%%############# BY CONTINENT
continents = {}
continents["Overall"] = diagnoses["Overall"].copy()
continents["NorthAmerica"] = diagnoses["Overall"].loc[(diagnoses["Overall"]["redcap_data_access_group"] == "bidmc") |(diagnoses["Overall"]["redcap_data_access_group"] == "mayo") | (diagnoses["Overall"]["redcap_data_access_group"] == "nih") | (diagnoses["Overall"]["redcap_data_access_group"] == "nyu") | (diagnoses["Overall"]["redcap_data_access_group"] == "utsw") | (diagnoses["Overall"]["redcap_data_access_group"] == "van") | (diagnoses["Overall"]["redcap_data_access_group"] == "cor") | (diagnoses["Overall"]["redcap_data_access_group"] == "stf")]
continents["SouthAmerica"] = diagnoses["Overall"].loc[diagnoses["Overall"]["redcap_data_access_group"] == "fleni"]
continents["Asia"] = diagnoses["Overall"].loc[diagnoses["Overall"]["redcap_data_access_group"] == "snu"]
continents["Europe"] = diagnoses["Overall"].loc[(diagnoses["Overall"]["redcap_data_access_group"] == "hcbm") | (diagnoses["Overall"]["redcap_data_access_group"] == "mui") | (diagnoses["Overall"]["redcap_data_access_group"] == "sms") | (diagnoses["Overall"]["redcap_data_access_group"] == "hcbs") | (diagnoses["Overall"]["redcap_data_access_group"] == "huc")]

for name, continent in continents.items():
    continent.name = name
del(name, continent)
#%%
ContinentsByDiagnosis = {"Overall" : pd.DataFrame(), "MSA" : pd.DataFrame(), "PD" : pd.DataFrame(), "DLB" : pd.DataFrame(), "PAF" : pd.DataFrame(), "RBD" : pd.DataFrame()}
def nPatients():
    for df in ContinentsByDiagnosis:
        ContinentsByDiagnosis[df] = pd.DataFrame(columns=["Overall", "North America", "South America", "Asia", "Europe"], index=["Patients (n)", "Sex: Male vs. Female", "Age Onset", "Disease Duration", "Global Disability Scale", "Orthostatic Hypotension", "RBD", "Erectile Dysfunction"])
    ContinentsByDiagnosis["MSA"] = pd.DataFrame(columns=["Overall", "North America", "South America", "Asia", "Europe"], index=["Patients (n)", "MSA-P vs. MSA-C", "Sex: Male vs. Female", "Age Onset", "Disease Duration", "Global Disability Scale", "Orthostatic Hypotension", "RBD", "Erectile Dysfunction"])
    for column, (name, df) in zip(ContinentsByDiagnosis["Overall"], continents.items()):
        ContinentsByDiagnosis["Overall"].at["Patients (n)", column] = df.shape[0]
    for (name, diagnosis) in islice(diagnoses.items(), 1, 6):
        for column, (name, continent) in zip(ContinentsByDiagnosis[diagnosis.name], continents.items()):
            ContinentsByDiagnosis[diagnosis.name].at["Patients (n)", column] = continent.loc[continent["diagnosis"] == diagnosis.name].shape[0]

nPatients()
#%%
continentsMSA = {}

continentsMSA["Overall"] = diagnoses["MSAC"].append(diagnoses["MSAP"])
continentsMSA["NorthAmerica"] = continentsMSA["Overall"].loc[(continentsMSA["Overall"]["redcap_data_access_group"] == "bidmc") |(continentsMSA["Overall"]["redcap_data_access_group"] == "mayo") | (continentsMSA["Overall"]["redcap_data_access_group"] == "nih") | (continentsMSA["Overall"]["redcap_data_access_group"] == "nyu") | (continentsMSA["Overall"]["redcap_data_access_group"] == "utsw") | (continentsMSA["Overall"]["redcap_data_access_group"] == "van") | (continentsMSA["Overall"]["redcap_data_access_group"] == "cor") | (continentsMSA["Overall"]["redcap_data_access_group"] == "stf")]
continentsMSA["SouthAmerica"] = continentsMSA["Overall"].loc[continentsMSA["Overall"]["redcap_data_access_group"] == "fleni"]
continentsMSA["Asia"] = continentsMSA["Overall"].loc[continentsMSA["Overall"]["redcap_data_access_group"] == "snu"]
continentsMSA["Europe"] = continentsMSA["Overall"].loc[(continentsMSA["Overall"]["redcap_data_access_group"] == "hcbm") | (continentsMSA["Overall"]["redcap_data_access_group"] == "mui") | (continentsMSA["Overall"]["redcap_data_access_group"] == "sms") | (continentsMSA["Overall"]["redcap_data_access_group"] == "hcbs") | (continentsMSA["Overall"]["redcap_data_access_group"] == "huc")]
for name, continent in continentsMSA.items():
    continent.name = name
del(name, continent)

def PvC():
    for column, (name, continent) in zip(ContinentsByDiagnosis["Overall"], continentsMSA.items()):
        n1 = continent.loc[continent["diagnosis"] == "MSAP"].shape[0]
        n2 = continent.loc[continent["diagnosis"] == "MSAC"].shape[0]
        string = str(n1) + "/" + str(n2)
        ContinentsByDiagnosis["MSA"].at["MSA-P vs. MSA-C", column] = string

PvC()
#%%
def count_vs(index, dataColumn, value1, value2):
    for column, (name, continent) in zip(ContinentsByDiagnosis["Overall"], continents.items()):
        nV1 = continent.loc[continent[dataColumn] == value1].shape[0]
        nV2 = continent.loc[continent[dataColumn] == value2].shape[0]
        nString = str(nV1) + "/" + str(nV2)
        ContinentsByDiagnosis["Overall"].at[index, column] = nString
    for (name, diagnosis) in islice(diagnoses.items(), 1, 6):
        for column, (name, continent) in zip(ContinentsByDiagnosis[diagnosis.name], continents.items()):
            if (ContinentsByDiagnosis[diagnosis.name].loc["Patients (n)", column]) == 0:
                ContinentsByDiagnosis[diagnosis.name][column] = 0
            else:
                nV1 = continent.loc[(continent["diagnosis"] == diagnosis.name) & (continent[dataColumn] == value1)].shape[0]
                nV2 = continent.loc[(continent["diagnosis"] == diagnosis.name) & (continent[dataColumn] == value2)].shape[0]
                nString = str(nV1) + "/" + str(nV2)
                ContinentsByDiagnosis[diagnosis.name].at[index, column] = nString

count_vs("Sex: Male vs. Female", "gender", 1, 2)
#%%
def mean_sd_String(index, variable):
    for column, (name, continent) in zip(ContinentsByDiagnosis["Overall"], continents.items()):
        mean = continent[variable].mean().round(1)
        std = continent[variable].std().round(1)
        string = str(mean) + " " + "$\pm$" + " " + str(std)
        ContinentsByDiagnosis["Overall"].at[index, column] = string
    for (name, diagnosis) in islice(diagnoses.items(), 1, 6):
        for column, (name, continent) in zip(ContinentsByDiagnosis[diagnosis.name], continents.items()):
            if (ContinentsByDiagnosis[diagnosis.name].loc["Patients (n)", column]) == 0:
                ContinentsByDiagnosis[diagnosis.name][column] = 0
            else:
                mean = continent.loc[continent["diagnosis"].str.contains(diagnosis.name), [variable]].mean().round(1).iloc[0]
                std = continent.loc[continent["diagnosis"].str.contains(diagnosis.name), [variable]].std().round(1).iloc[0]
                string = str(mean) + " " + "$\pm$" + " " + str(std)
                ContinentsByDiagnosis[diagnosis.name].at[index, column] = string

mean_sd_String("Age Onset", "ageAtOnset")
mean_sd_String("Disease Duration", "durationOfSymptoms")
mean_sd_String("Global Disability Scale", "umsarsdis1")
#%%
def nCounts(index, dataColumn):
    for column, (name, continent) in zip(ContinentsByDiagnosis["Overall"], continents.items()):
        ContinentsByDiagnosis["Overall"].at[index, column] = continent.loc[continent[dataColumn] == 1].shape[0]
    for (name, diagnosis) in islice(diagnoses.items(), 1, 6):
        for column, (name, continent) in zip(ContinentsByDiagnosis[diagnosis.name], continents.items()):
            if (ContinentsByDiagnosis[diagnosis.name].loc["Patients (n)", column]) == 0:
                ContinentsByDiagnosis[diagnosis.name][column] = 0
            elif np.isnan(continent.loc[(continent["diagnosis"] == diagnosis.name) & (continent[dataColumn] == 1)].shape[0]) == True:
                ContinentsByDiagnosis[diagnosis.name].at[index, column] = 0
            else:
                ContinentsByDiagnosis[diagnosis.name].at[index, column] = continent.loc[(continent["diagnosis"] == diagnosis.name) & (continent[dataColumn] == 1)].shape[0]

nCounts("Orthostatic Hypotension", "hasOH")
nCounts("RBD", "screen_rbd")
nCounts("Erectile Dysfunction", "erect_dys_check")
#%%
def table_to_latex(dict):
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
    for df, diagnosis in zip(dict, ["All Diagnoses", "Multiple System Atrophy", "Parkinson's Disease", "Dementia with Lewy Bodies", "Pure Autonomic Failure", "REM Behavior Disorder"]):
        table_name = df + ".tex"
        f = open(table_name, 'w')
        f.write(beginningtex.replace("mycaption", diagnosis + " by continent"))
        f.write(dict[df].to_latex(index=True, column_format="lccccc", escape= False))
        f.write(endtex)
        f.close()
#%%
table_to_latex(ContinentsByDiagnosis)
