import numpy as np
import pandas as pd
from scipy import stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison
import statsmodels.formula.api as smf


#### two-way ANOVA test ####
## source
df = pd.read_csv("/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_transverstemporal.csv")
mask = df['subject ID'].isin([697, 750, 853, 841]) # 841 is just for fun
df = df[~mask]
selected_columns = ['subject ID', 'Stimulus', 'Hemisphere', 'area inhibition (30ms)', 'EOG area inhibition']
df = df[selected_columns]

conditions = [
    df['Stimulus'].str.endswith('_i0'),
    df['Stimulus'].str.endswith('_i60'),
    df['Stimulus'].str.endswith('_i120'),
    df['Stimulus'].str.endswith('_i240')
]
choices = ['0', '60', '120', '240']
df['Inter Stimulus Interval'] = np.select(conditions, choices, default=np.nan)
mask = df.apply(lambda row: any(['nan' in str(cell) for cell in row]), axis=1)
df = df[~mask]
df['Pulse level'] = df['Stimulus'].apply(lambda x: '60' if x.startswith('GP60') else ('70' if x.startswith('GP70') else None))
df = df.rename(columns={'subject ID': 'ID', 'area inhibition (30ms)': 'II',
                        'EOG area inhibition': 'EOG_II',
                        'Inter Stimulus Interval': 'ISI',
                        'Pulse level': 'PL'})
df_rh = df[df['Hemisphere'] == 'rh']
df_lh = df[df['Hemisphere'] == 'lh']

## checking for missing values
assert not sum(df.isnull().sum()), "There are missing values" 

## removing outliers
# thr = 3
# z_scores_rh = stats.zscore(df_rh['II'])
# z_scores_lh = stats.zscore(df_lh['II'])
# z_scores_eog = stats.zscore(df_rh['EOG_II'])

# df_rh = df_rh[~(abs(z_scores_rh) > thr)]
# df_lh = df_lh[~(abs(z_scores_lh) > thr)]
# df_eog = df_rh[~(abs(z_scores_eog) > thr)]

## performing two-way ANOVA
model_rh = ols(formula='II ~ ISI + PL + ISI:PL',
            data=df_rh,
            drop_cols=['ID', 'Stimulus', 'Hemisphere', 'EOG_II']).fit()
model_lh = ols(formula='II ~ ISI + PL + ISI:PL',
            data=df_lh,
            drop_cols=['ID', 'Stimulus', 'Hemisphere', 'EOG_II']).fit()
model_eog = ols(formula='EOG_II ~ ISI + PL + ISI:PL',
            data=df_rh,
            drop_cols=['ID', 'Stimulus', 'Hemisphere', 'II']).fit()

## performing LLM
result_llm_rh = smf.mixedlm("II ~ ISI + PL", df_rh, groups=df_rh.index).fit()
result_llm_lh = smf.mixedlm("II ~ ISI + PL", df_lh, groups=df_lh.index).fit()
# result_llm_eog = smf.mixedlm("II ~ ISI + PL", df_eog, groups=df_eog.index).fit()

## checking normality with Shapiro-Wilk test
for model, title in zip([model_rh, model_lh, model_eog], ["rh", "lh", "eog"]):
    residuals = model.resid
    shapiro_test_residuals = stats.shapiro(residuals)
    # print(f"Shapiro-Wilk test for residuals in {title}:", shapiro_test_residuals)

table_rh = sm.stats.anova_lm(model_rh, typ=2)
table_lh = sm.stats.anova_lm(model_lh, typ=2)
table_eog = sm.stats.anova_lm(model_eog, typ=2)

## ANOVA repeated measure
aovrm2way_rh = AnovaRM(df_rh, "II", "ID", within=["ISI", "PL"]).fit()
aovrm2way_lh = AnovaRM(df_lh, "II", "ID", within=["ISI", "PL"]).fit()
aovrm2way_eog = AnovaRM(df_rh, "EOG_II", "ID", within=["ISI", "PL"]).fit()

## tukey multi comparison
tukey_rh = MultiComparison(data=df_rh['II'], groups=df_rh['ISI']).tukeyhsd() # could be changed to PL
tukey_lh = MultiComparison(data=df_lh['II'], groups=df_lh['ISI']).tukeyhsd()
tukey_eog = MultiComparison(data=df_rh['EOG_II'], groups=df_rh['ISI']).tukeyhsd()

#### on sensor spcae ####

## sensor
df = pd.read_csv("/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_sensors.csv")
mask = df['subject ID'].isin([697, 750, 853, 841])
df = df[~mask]
selected_columns = ['subject ID', 'Stimulus', 'Ch_name', 'area inhibition (30ms)']
df = df[selected_columns]

conditions = [
    df['Stimulus'].str.endswith('_i0'),
    df['Stimulus'].str.endswith('_i60'),
    df['Stimulus'].str.endswith('_i120'),
    df['Stimulus'].str.endswith('_i240')
]
choices = ['0', '60', '120', '240']
df['Inter Stimulus Interval'] = np.select(conditions, choices, default=np.nan)
mask = df.apply(lambda row: any(['nan' in str(cell) for cell in row]), axis=1)
df = df[~mask]
df['Pulse level'] = df['Stimulus'].apply(lambda x: '60' if x.startswith('GP60') else ('70' if x.startswith('GP70') else None))
df = df.rename(columns={'subject ID': 'ID', 'area inhibition (30ms)': 'II',
                        'Inter Stimulus Interval': 'ISI',
                        'Pulse level': 'PL'})
df_rh = df[df['Ch_name'] == 'MEG1332']
df_lh = df[df['Ch_name'] == 'MEG0242']

## removing outliers
# thr = 3
# z_scores_rh = stats.zscore(df_rh['II'])
# z_scores_lh = stats.zscore(df_lh['II'])

# df_rh = df_rh[~(abs(z_scores_rh) > thr)]
# df_lh = df_lh[~(abs(z_scores_lh) > thr)]

## performing two-way ANOVA and tukey test
model_rh = ols(formula='II ~ C(ISI) + C(PL) + C(ISI):C(PL)',
            data=df_rh,
            drop_cols=['ID', 'Stimulus', 'Ch_name', 'EOG_II']).fit()
model_lh = ols(formula='II ~ C(ISI) + C(PL) + C(ISI):C(PL)',
            data=df_lh,
            drop_cols=['ID', 'Stimulus', 'Ch_name', 'EOG_II']).fit()

table_rh = sm.stats.anova_lm(model_rh, typ=2)
table_lh = sm.stats.anova_lm(model_lh, typ=2)

## ANOVA repeated measure
aovrm2way_rh = AnovaRM(df_rh, "II", "ID", within=["ISI", "PL"]).fit()
aovrm2way_lh = AnovaRM(df_lh, "II", "ID", within=["ISI", "PL"]).fit()

tukey_rh = MultiComparison(data=df_rh['II'], groups=df_rh['ISI']).tukeyhsd() # could be changed to PL
tukey_lh = MultiComparison(data=df_lh['II'], groups=df_lh['ISI']).tukeyhsd()

#### ANOVA pulse only ####
## preparing the dataframe for anova pulse only
df = pd.read_csv("/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_transverstemporal.csv")
# remove some subjects
mask = df['subject ID'].isin([697, 750, 853, 841]) # 841 is just for fun
df = df[~mask]
# take only pulse only
mask_po = df["Stimulus"].isin(["PO60_75", "PO60_80", "PO60_85", "PO60_90", "PO60_95",
                                "PO70_75", "PO70_80", "PO70_85", "PO70_90", "PO70_95"])
df = df[mask_po]
# create two new columns PL and BBN
def check_BBN(string):
    if 'PO60' in string:
        return 60
    elif 'PO70' in string:
        return 70
def check_PL(string):
    return string[-2:]   
df['BBN'] = df['Stimulus'].apply(check_BBN)
df['PL'] = df['Stimulus'].apply(check_PL)
# rename latency columns
df = df.rename(columns={"EOG peak latency": "EOG_peak_latency", "EOG peak": "EOG_peak",
                        "peak value (30ms)": "peak_value_30ms", "subject ID": "ID"})
df_eog = df[df["Hemisphere"]=="rh"]
df_rh = df[df["Hemisphere"]=="rh"]
df_lh = df[df["Hemisphere"]=="lh"]

# eog latency anova
model_eog_lat = ols(formula='EOG_peak_latency ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_eog).fit()
table_eog_lat = sm.stats.anova_lm(model_eog_lat, typ=2)
aovrm2way_eog_lat = AnovaRM(df_eog, "EOG_peak_latency", "ID", within=["PL", "BBN"]).fit()

# lh latency anova
model_lh_lat = ols(formula='latency ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_lh).fit()
table_lh_lat = sm.stats.anova_lm(model_lh_lat, typ=2)
aovrm2way_lh_lat = AnovaRM(df_lh, "latency", "ID", within=["PL", "BBN"]).fit()

# rh latency anova
model_rh_lat = ols(formula='latency ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_rh).fit()
table_rh_lat = sm.stats.anova_lm(model_rh_lat, typ=2)
aovrm2way_rh_lat = AnovaRM(df_rh, "latency", "ID", within=["PL", "BBN"]).fit()

# eog peak amplitude
model_eog_peak = ols(formula='EOG_peak ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_eog).fit()
table_eog_peak = sm.stats.anova_lm(model_eog_peak, typ=2)
aovrm2way_eog_peak = AnovaRM(df_eog, "EOG_peak", "ID", within=["PL", "BBN"]).fit()

# lh peak amplitude
model_lh_peak = ols(formula='peak_value_30ms ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_lh).fit()
table_lh_peak = sm.stats.anova_lm(model_lh_peak, typ=2)
aovrm2way_lh_lat = AnovaRM(df_lh, "peak_value_30ms", "ID", within=["PL", "BBN"]).fit()

# rh peak amplitude
model_rh_peak = ols(formula='peak_value_30ms ~ C(PL) + C(BBN) + C(PL):C(BBN)',
            data=df_rh).fit()
table_rh_peak = sm.stats.anova_lm(model_rh_peak, typ=2)
aovrm2way_rh_lat = AnovaRM(df_rh, "peak_value_30ms", "ID", within=["PL", "BBN"]).fit()


#### ANOVA Gap Only #### 

## preparing the dataframe for anova gap only
df = pd.read_csv("/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_transverstemporal.csv")
# remove some subjects
mask = df['subject ID'].isin([697, 750, 853, 841]) # 841 is just for fun
df = df[~mask]
# take only pulse only
mask_po = df["Stimulus"].isin(["GO_60", "GO_70"])
df = df[mask_po]
# create two new columns PL and BBN
def check_BBN(string):
    if '60' in string:
        return 60
    elif '70' in string:
        return 70
df['BBN'] = df['Stimulus'].apply(check_BBN)
# rename columns
df = df.rename(columns={"peak value (30ms)": "peak_value_30ms", "subject ID": "ID"})

df_rh_60 = df[(df["Hemisphere"]=="rh") & (df["BBN"]==60)]
df_lh_60 = df[(df["Hemisphere"]=="lh") & (df["BBN"]==60)]
df_rh_70 = df[(df["Hemisphere"]=="rh") & (df["BBN"]==70)]
df_lh_70 = df[(df["Hemisphere"]=="lh") & (df["BBN"]==70)]

print(f_oneway(df_rh_60["peak_value_30ms"], df_lh_60["peak_value_30ms"]))
print(f_oneway(df_rh_70["peak_value_30ms"], df_lh_70["peak_value_30ms"]))

##
print(f_oneway(df_rh_60["EOG peak"], df_rh_70["EOG peak"]))
print(f_oneway(df_lh_60["peak_value_30ms"], df_lh_70["peak_value_30ms"]))
print(f_oneway(df_rh_60["peak_value_30ms"], df_rh_70["peak_value_30ms"]))