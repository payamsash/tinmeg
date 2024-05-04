import mne
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib
from tqdm.notebook import tqdm
from itertools import product
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA


# Create epochs dictionary (some needs concatenating)
epochs_folder = '/Users/payamsadeghishabestari/KI_MEG/epochs/tinmeg1'
epochs_file = {}
for f in sorted(os.listdir(epochs_folder)):
    file = os.path.join(epochs_folder, f)
    if file.endswith("-epo.fif"):
        epochs_file[f'{file[-11:-8]}'] = file

# Create epochs dictionary (some needs concatenating)
epochs_folder = '/Users/payamsadeghishabestari/KI_MEG/epochs/tinmeg1'
epochs_file = {}
for f in sorted(os.listdir(epochs_folder)):
    file = os.path.join(epochs_folder, f)
    if file.endswith("-epo.fif") and '697' not in file and '750' not in file and '853' not in file:
        epochs_file[f'{file[-11:-8]}'] = file

# compute evoked objects, and making grand average dictionary
evs = []
for ep_f in tqdm(list(epochs_file.values())):
    evs.append(mne.read_epochs(fname=ep_f, verbose=False).average(picks=['meg', 'eog'], by_event_type=True))

grnd_ev_dict = {}
for stim_idx, stim in enumerate(list(events_dict_tinmeg1.keys())):
    evs_stim = []
    for ev in evs:
        evs_stim.append(ev[stim_idx])
    grnd_ev_dict[stim] = evs_stim

grand_ev_dict = {}
for stim in list(grnd_ev_dict.keys()):
    grand_ev_dict[stim] = mne.grand_average(grnd_ev_dict[stim])

# separate chanels on left and right
info_ch = evs[0][0].info['chs']
meg_chs_right = []; meg_chs_left = []
grad_chs_right = []; grad_chs_left = []
for i in range(len(info_ch)):
    if info_ch[i]['unit'] == 112: # meg code
        if info_ch[i]['loc'][0] > 0:
            meg_chs_right.append(info_ch[i]['ch_name'])
        if info_ch[i]['loc'][0] < 0:
            meg_chs_left.append(info_ch[i]['ch_name'])
    if info_ch[i]['unit'] == 201: # grad code
        if info_ch[i]['loc'][0] > 0:
            grad_chs_right.append(info_ch[i]['ch_name'])
        if info_ch[i]['loc'][0] < 0:
            grad_chs_left.append(info_ch[i]['ch_name'])

# select the left/ channels with largest ptp amplitude
ev_data_left = grand_ev_dict['PO60_70'].get_data(picks=grad_chs_left)
ev_data_right = grand_ev_dict['PO60_70'].get_data(picks=grad_chs_right)
max_values = []
for ch_idx in range(len(ev_data_left)):
    max_values.append(ev_data_left[ch_idx][50:150].max())
ch_max_left = grad_chs_left[np.argmax(np.array(max_values))]
max_values = []
for ch_idx in range(len(ev_data_right)):
    max_values.append(ev_data_right[ch_idx][50:150].max())
ch_max_right = grad_chs_right[np.argmax(np.array(max_values))]

#### Making a dataframe for it ####

# making big dataframe for transverstemporal rh/lh
my_dict = {'subject ID': [], 'Stimulus': [], 'Ch_name': [],
        'peak value (30ms)': [], 'peak value (100ms)': [], 'area value (30ms)': [],
        'area value (100ms)': [], 'amplitude inhibition (30ms)': [],
        'area inhibition (30ms)': [], 'amplitude inhibition (100ms)': [],
        'area inhibition (100ms)': []} 

events = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95',
        'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95',
        'GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240',
        'GP70_i0', 'GP70_i60', 'GP70_i120', 'GP70_i240',
        'GO_60', 'GO_70']

subjects = ['539', '756', '832', '835', '836', '838', '839', '840',
        '841', '842', '844', '845', '847', '849','850', '852',
        '856', '857', '858', '859', '861', '862', '863'] 

chs = [ch_max_left, ch_max_right]
combinations = product(subjects, events, chs)

for subject, stim, ch in tqdm(combinations):
                
        # reading epoch file 
        fname = epochs_file[subject]
        ep = mne.read_epochs(fname=fname, preload=True, verbose=False)[stim]
        ep_data = np.squeeze(ep.get_data(picks=ch)).mean(axis=0) 

        # computing some params  
        ep_peak_100ms = ep_data[87:114].max()
        ep_peak_30ms = ep_data[96:105].max()
        ep_area_100ms = ep_data[87:114].sum()
        ep_area_30ms = ep_data[96:105].sum()
        
        if '60' in stim and '70' not in stim:
                ep_stn = mne.read_epochs(fname=fname, preload=True, verbose=False)['PO60_90']
                ep_stn_data = np.squeeze(ep_stn.get_data(picks=ch)).mean(axis=0)
                ep_stn_peak_100ms = ep_stn_data[87:114].max()
                ep_stn_peak_30ms = ep_stn_data[96:105].max()
                ep_stn_area_100ms = ep_stn_data[87:114].sum()
                ep_stn_area_30ms = ep_stn_data[96:105].sum()
        if '70' in stim:
                ep_stn = mne.read_epochs(fname=fname, preload=True, verbose=False)['PO70_90']
                ep_stn_data = np.squeeze(ep_stn.get_data(picks=ch)).mean(axis=0)
                ep_stn_peak_100ms = ep_stn_data[87:114].max()
                ep_stn_peak_30ms = ep_stn_data[96:105].max()
                ep_stn_area_100ms = ep_stn_data[87:114].sum()
                ep_stn_area_30ms = ep_stn_data[96:105].sum()

        # computing inhibition indexes 
        my_dict['amplitude inhibition (30ms)'].append((1 - (ep_peak_30ms / ep_stn_peak_30ms)) * 100)
        my_dict['area inhibition (30ms)'].append((1 - (ep_area_30ms / ep_stn_area_30ms)) * 100)
        my_dict['amplitude inhibition (100ms)'].append((1 - (ep_peak_100ms / ep_stn_peak_100ms)) * 100)
        my_dict['area inhibition (100ms)'].append((1 - (ep_area_100ms / ep_stn_area_100ms)) * 100)

        # putting in the dictionary
        my_dict['subject ID'].append(subject)
        my_dict['Stimulus'].append(stim)
        my_dict['Ch_name'].append(ch)
        my_dict['peak value (30ms)'].append(ep_peak_30ms)
        my_dict['peak value (100ms)'].append(ep_peak_100ms)
        my_dict['area value (30ms)'].append(ep_area_30ms)
        my_dict['area value (100ms)'].append(ep_area_100ms)

df = pd.DataFrame(my_dict)
# save it
df.to_csv('/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_sensors.csv')

#### plotting ####

df = pd.read_csv('/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_sensors.csv')

# Supplementary figure A
order_1 = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']
order_2 = ['PO60_70', 'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95']
mask = df['Stimulus'].isin(order_2)
df1 = df[mask]
fig, ax = plt.subplots(1, 1, figsize=(10,4))
palette_color = ['#1f77b4', '#d62728'] 
sns.boxplot(data=df1, x='Stimulus', y='peak value (30ms)', hue='Ch_name', width=0.8, fill=False, gap=.1, linewidth=2,
            saturation=0.75, palette=palette_color, order=order_2, ax=ax)
sns.stripplot(data=df1, x='Stimulus', y='peak value (30ms)', hue='Ch_name',
            dodge=True, size=3, palette=palette_color, order=order_2, ax=ax, legend=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper left')
ax.set_yticks(np.array([0, 0.5, 1, 1.5])*1e-11)

# supplementary figure B
fig, axs = plt.subplots(1, 1, figsize=(7, 3))
time_array = np.linspace(-300, 300, 151)
stims = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']
# stims = ['PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95']
colors = ['#1f77b4', '#d62728']
color = colors[0]
lw = 0.5
for stim in stims[:-1]:
    axs.plot(time_array, abs(grand_ev_dict[stim].get_data(picks=ch_max_left)[0] * 1e15),
            linewidth=lw, label=stim, color=color)
    lw += 0.4
for stim in stims[-1:]:
    axs.plot(time_array, abs(grand_ev_dict[stim].get_data(picks=ch_max_left)[0] * 1e15),
            linewidth=lw, label=stim, color='k')
    
axs.axvspan(50, 150, alpha=0.4, color='lightgrey')
# axs.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.1, 0.6, 0.6))
axs.spines['top'].set_visible(False); axs.spines['right'].set_visible(False)
for i in [-200, -100, 100, 200, 300]:
    axs.vlines(i, -1000, 10000, colors='black',linestyles=':', linewidth=0.5)
axs.vlines(0, -1000, 10000, colors='black',linestyles='--')
axs.set_ylabel(f'fT/cm')
axs.set_xlabel(f'Time (ms)')
axs.set_ylim([-1000, 10000])

# supplementary figure B
info = grand_ev_dict['PO60_70'].pick('grad').info
kwargs = dict(eeg=False, coord_frame="mag")
n_chs = len(info['ch_names'])
sensor_colors = np.zeros(shape=(n_chs, 4)) + matplotlib.colors.to_rgba_array('grey', alpha=0.3)
left_idx = info['ch_names'].index(ch_max_left)
right_idx = info['ch_names'].index(ch_max_right)
sensor_colors[left_idx] = matplotlib.colors.to_rgba_array('#1f77b4', alpha=None)
sensor_colors[right_idx] = matplotlib.colors.to_rgba_array('#d62728', alpha=None)

fig = mne.viz.create_3d_figure((600, 600), bgcolor=(255, 255, 255))
mne.viz.plot_alignment(info=info, surfaces='auto', coord_frame='auto',
                    meg='sensors', eeg=False, ecog=True, fig=fig,
                    dbs=False, interaction='terrain', sensor_colors=sensor_colors)


#### canonical correlation plot ####
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

fig, axs = plt.subplots(2, 4, figsize=(11, 5))
fig.subplots_adjust(hspace=0.5)
axes = list(product(range(2), range(4)))
stims = ["GP60_i0", "GP60_i60", "GP60_i120", "GP60_i240",
        "GP70_i0", "GP70_i60", "GP70_i120", "GP70_i240"]
cmap = plt.get_cmap("Set1")
df_rh = df[df['Hemisphere'] == 'rh']
df_lh = df[df['Hemisphere'] == 'lh']

for (i , j), stim in zip(axes, stims):
    # create the dataframe
    df1 = df_rh[df_rh["Stimulus"]==stim]
    df2 = df_lh[df_lh["Stimulus"]==stim]
    new_df_dict = {"Stimulus": [stim]*len(df1),
                    "right_II": np.array(df1["II"]),
                    "left_II": np.array(df2["II"]),
                    "EOG_II": np.array(df1["EOG_II"])}
    df_both = pd.DataFrame(new_df_dict)
    df_both = df_both[(df_both["right_II"]>0) & (df_both["left_II"]>0) & (df_both["EOG_II"]>0)]

    ## CC analysis
    X = df_both[["left_II", "right_II"]]
    Y = df_both["EOG_II"]
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)

    ## plotting
    scatter_kws = {"s": 10, "color": cmap.colors[j]}
    line_kws = {"linestyle": "--", "linewidth": 1, "color": "k"}
    sns.regplot(data=df_both, x=X_c, y=Y_c, ax=axs[i][j], ci=95, scatter_kws=scatter_kws,
                line_kws=line_kws)
    axs[i][j].set_xlabel("")
    axs[i][j].set_ylabel("")
    axs[i][j].spines[['right', 'top']].set_visible(False)
    axs[i][j].set_xlim([-2.5, 2.5])
    axs[i][j].set_ylim([-2.5, 2.5])

    # to print p and r values
    sm_x = sm.add_constant(X_c) 
    model = sm.GLS(Y_c, sm_x)
    results = model.fit()
    axs[i][j].set_title(f"R_const:{round(results.params[0], 3)}, R_ii:{round(results.params[1], 3)}\n p_const:{round(results.pvalues[0], 3)}, p_ii:{round(results.pvalues[1], 3)}", fontsize=6)


axs[0][0].set_ylabel("EOG Inhibition")
axs[1][0].set_ylabel("EOG Inhibition")
# for i, title, xlabel in zip(range(4), ["0 ms", "60 ms", "120 ms", "240 ms"], ["Inhibition Index"] * 4):
#     axs[0][i].set_title(title)
#     axs[1][i].set_xlabel(xlabel)

axs[0][3].yaxis.set_label_position("right")
axs[1][3].yaxis.set_label_position("right")

axs[0][3].set_ylabel("Noise Level 60", rotation=270)
axs[1][3].set_ylabel("Noise Level 70", rotation=270)