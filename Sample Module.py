## Module including a function to create and clean a pandas DataFrame from a direct export from REDCap using API. 

from redcap import Project, RedcapError
import numpy as np
import pandas as pd
import math

def createDF(visitName):
    project = Project('https://openredcap.nyumc.org/apps/redcap/api/', '################################')
    all = project.export_records(format='df', df_kwargs={'index_col': project.field_names[0]})
    all["docode"] = pd.to_datetime(all["docode"])
    all["consentstatusdt"] = pd.to_datetime(all["consentstatusdt"])

    for column in ["gender", "consentstatusdt", "docode", "diag_entry", "hxpisymptoms"]:
        x = all[(all.redcap_event_name == "1_entry_visit_arm_1")][column]
        all[column] = all.index.map(x)
    sub = all[all["redcap_event_name"] == visitName]
    sub = sub.rename(columns={"diag_entry": "diagnosis"})
    sub["ageAtEnrollment"] = sub['consentstatusdt'].sub(sub['docode'], axis=0) / np.timedelta64(1, 'Y')
    sub["yearOfBirth"] = sub["docode"].apply(lambda x: x.year)

    sub = sub.rename(
        columns={"afttilheadup3sys": "upSBP", "afttiltsupine0sys": "supSBP", "afttiltheadup3dias": "upDBP",
                 "afttiltsupine0dias": "supDBP", "afttiltheadup3hr": "upHR", "afttiltsupine0hr": "supHR",
                 "catuprightne": "upNE", "catssupinene": "supNE"})

    for column in ["hxpisymptoms", "upSBP", "supSBP", "upDBP", "supDBP", "upHR", "supHR",
                   "upNE", "supNE", "aftvalsalvaratio", "aftrrdeeppacedeiratio", "motor_score", "non_motor",
                   "mood_score", "total_qol_score", "umsars1scocalc", "umsars2scocalc", "umsarsadl10", "umsarsadl12",
                   "umsarsdis1", "moca_sco", "upsit"]:
        for index in sub.index:
            try:
                sub.at[index, column] = float(sub.at[index, column])
            except ValueError:
                sub.at[index, column] = np.nan

    sub["ageAtOnset"] = sub["hxpisymptoms"] - sub["yearOfBirth"]
    for value in sub["ageAtOnset"]:
        if value < 30:
            sub["ageAtOnset"].replace(value, np.nan, inplace=True)


    def cleanColumn(column1, column2):
        column1Clean = column1 + "clean"
        column2Clean = column2 + "clean"
        for index in sub.index:
            if (math.isnan(sub.at[index, column1]) == False) and (
                    math.isnan(sub.at[index, column2]) == False):
                sub.at[index, column1Clean] = sub.at[index, column1]
                sub.at[index, column2Clean] = sub.at[index, column2]
            else:
                sub.at[index, column1Clean] = np.nan
                sub.at[index, column2Clean] = np.nan

    cleanColumn("ageAtOnset", "ageAtEnrollment")

    sub["durationOfSymptoms"] = np.nan
    for index in sub.index:
        for i, add in zip(["1_entry_visit_arm_1", "2_followup_year_1_arm_1", "3_followup_year_2_arm_1", "4_followup_year_3_arm_1", "5_followup_year_4_arm_1", "6_followup_year_5_arm_1", "7_followup_year_6_arm_1", "8_followup_year_7_arm_1", "9_followup_year_8_arm_1", "10_followup_year_9_arm_1", "11_followup_year_10_arm_1"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            if sub.at[index, "redcap_event_name"] == i:
                sub.at[index, "durationOfSymptoms"] = sub.at[index,"ageAtEnrollment"] - sub.at[index,"ageAtOnset"] + add
    sub = sub.drop("redcap_event_name", axis=1)
    cleanColumn("upSBP", "supSBP")
    sub["cSBP"] = sub["upSBPclean"] - sub["supSBPclean"]
    cleanColumn("upDBP", "supDBP")
    sub["cDBP"] = sub["upDBPclean"] - sub["supDBPclean"]
    cleanColumn("upHR", "supHR")
    sub["cHR"] = sub["upHRclean"] - sub["supHRclean"]
    cleanColumn("upNE", "supNE")
    sub["cNE"] = sub["upNEclean"] - sub["supNEclean"]
    sub["cHR/cSBP"] = abs(sub["cHR"] / sub["cSBP"])

    def HRdivSBP(df):
        if df["cHR/cSBP"] < 0.5:
            return 1
        elif (df["cHR/cSBP"] >= 0.0):
            return 2
        else:
            return None

    sub["nOHratio"] = sub.apply(HRdivSBP, axis=1)

    def OH_type(df):
        if (df["cSBP"] <= -30) or (df["cDBP"] <= -15):
            return 1
        elif (-30 < df["cSBP"] <= -20) or (-15 < df["cDBP"] <= -10):
            return 2
        if (np.isnan(df["cSBP"]) == False) or (np.isnan(df["cDBP"]) == False):
            return 3
        else:
            return None

    sub["typeOH"] = sub.apply(OH_type, axis=1)

    def hasOH(df):
        if (df["typeOH"] == 1) or (df["typeOH"] == 2):
            return 1
        elif df["typeOH"] == 3:
            return 2
        else:
            return None

    sub["hasOH"] = sub.apply(hasOH, axis=1)

    def hasSH(df):
        if (df["supSBP"] > 140) or (df["supDBP"] > 90):
            return 1
        elif (np.isnan(df["supSBP"]) == False) or (np.isnan(df["supDBP"]) == False):
            return 2
        else:
            return None

    sub["hasSH"] = sub.apply(hasSH, axis=1)

    def UMSARSurinaryFunc(df):
        if (df["umsarsadl10"] >= 2):
            return 1
        elif (df["umsarsadl10"] < 2):
            return 2
        else:
            return None

    sub["UMSARSurinaryFunc"] = sub.apply(UMSARSurinaryFunc, axis=1)

    def UMSARSbowelFunc(df):
        if (df["umsarsadl12"] >= 1):
            return 1
        elif (df["umsarsadl12"] < 1):
            return 2
        else:
            return None

    sub["UMSARSbowelFunc"] = sub.apply(UMSARSbowelFunc, axis=1)

    def MOCA_23(df):
        if (df["moca_sco"] < 23):
            return 1
        elif (df["moca_sco"] >= 23):
            return 2
        else:
            return None

    sub["MOCA_23"] = sub.apply(MOCA_23, axis=1)

    def UPSIT_18(df):
        if (df["upsit"] < 18):
            return 1
        elif (df["upsit"] >= 18):
            return 2
        else:
            return None

    sub["UPSIT_18"] = sub.apply(UPSIT_18, axis=1)




    sub.diagnosis.replace([1, 2], "PAF", inplace=True)
    sub.diagnosis.replace(3, "MSAC", inplace=True)
    sub.diagnosis.replace(4, "MSAP", inplace=True)
    sub.diagnosis.replace([5, 6], "PD", inplace=True)
    sub.diagnosis.replace(7, "DLB", inplace=True)
    sub.diagnosis.replace(8, "RBD", inplace=True)
    sub.diagnosis.replace(9, np.nan, inplace=True)
    sub.dropna(subset=["diagnosis"], inplace = True)

    sub.loc[:, "prog_disease": "speakact_dreams"].replace([2, 3], 2, inplace=True)
    return sub
