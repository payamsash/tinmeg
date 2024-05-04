import mne
import numpy as np
import pandas as pd
import seaborn as sns
import os
import os.path as op
import matplotlib.pyplot as plt
from mne.datasets import fetch_fsaverage
from scipy import stats as stats


#### load dataframe and remove three subjects ####
df = pd.read_csv('/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg1_transverstemporal.csv') 
mask = df['subject ID'].isin([697, 750, 853, 841])
df = df[~mask]


#### figure 1 ####

# plotting Figure 1 a
fig, ax = plt.subplots(1, 1, figsize=(4,4))
color='grey'
order_1 = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']
order_2 = ['PO60_70', 'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95']
df1 = df[df['Hemisphere']=='rh']
sns.boxplot(data=df1, x='Stimulus', y='EOG area (30ms)', width=0.8, fill=False, gap=.1, linewidth=2,
            saturation=0, color=color, order=order_2, ax=ax)
color='#9467bd' # #ff7f0e
sns.stripplot(data=df1, x='Stimulus', y='EOG area (30ms)',
            dodge=False, size=3, color=color, order=order_2, ax=ax)
ax.set_ylim([0, 1100])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# plotting Figure 1 b
order_1 = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']
order_2 = ['PO60_70', 'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95']
mask = df['Stimulus'].isin(order_1)
df1 = df[mask]
fig, ax = plt.subplots(1, 1, figsize=(10,4))
palette_color = ['#1f77b4', '#d62728'] 
sns.boxplot(data=df1, x='Stimulus', y='peak value (30ms)', hue='Hemisphere', width=0.8, fill=False, gap=.1, linewidth=2,
            saturation=0.75, palette=palette_color, order=order_1, ax=ax)
sns.stripplot(data=df1, x='Stimulus', y='peak value (30ms)', hue='Hemisphere',
            dodge=True, size=3, palette=palette_color, order=order_1, ax=ax, legend=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper left')

# plotting Figure 1 c
fig, ax = plt.subplots(figsize=(11, 4))
colors = sns.color_palette('Set1')[1:5]
# colors = np.array(sns.color_palette('Set1'))[[1, 2, 6, 7, 3, 0, 8],:]
data = [19, 1, 2, 1]
# data = [11, 4, 2, 2, 2, 1, 1]
ingredients = ['transverstemporal-rh', 'bankssts-rh', 'entorhinal-rh', 'temporalpole-rh']
# ingredients = ['transverstemporal-lh', 'bankssts-lh', 'rostralanteriorcingulate-lh', 'paracentral-lh',
#                'entorhinal-lh', 'posteriorcingulate-lh', 'lateralorbitofrontal-lh']
def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%"
wedges, autotexts = ax.pie(data, textprops=dict(color="w"), colors=colors)
legend_names = [f'{name} ({round(number/23*100, 1)}%)' for name, number in zip(ingredients, data)]
ax.legend(wedges, legend_names, loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=12)

# figure 1 d (needs to run previous parts)
grand_ev_dict['PO60_90'].plot_topomap(times=0.1, time_unit='ms', contours=6)

# figure 1 e
directory = '/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg1_morphed'
stc_files_list_rh = {'PO60_70': [], 'PO60_75': [], 'PO60_80': [],
                'PO60_85': [], 'PO60_90': [], 'PO60_95': []}
stc_files_list_lh = {'PO60_70': [], 'PO60_75': [], 'PO60_80': [],
                'PO60_85': [], 'PO60_90': [], 'PO60_95': []}
events = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']

brain_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc')[:-1] # 68 labels
fs_dir = fetch_fsaverage(verbose=False)
src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(src_fname, verbose=False)

for event in events:
    for filename in sorted(os.listdir(directory)): 
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and f.endswith("-lh.stc") and event in f: # or -rh
                stc = mne.read_source_estimate(fname=f, subject='fsaverage')
                rh_data = stc.extract_label_time_course(brain_labels[-1], src, mode='mean', verbose=False)
                lh_data = stc.extract_label_time_course(brain_labels[-2], src, mode='mean', verbose=False)
                stc_files_list_rh[event].append(rh_data)
                stc_files_list_lh[event].append(lh_data)

fig, ax = plt.subplots(1, 1, figsize=(7,3))
lw = 0.5
colors = ['#1f77b4', '#d62728']
for event in events[:-1]:
    data = np.squeeze(np.array(stc_files_list_lh[event])).mean(axis=0)
    ax.plot(np.linspace(-300, 300, 151), data.T, label=event, linewidth=lw, color=colors[0])
    lw += 0.4
for event in events[-1:]:
    data = np.squeeze(np.array(stc_files_list_lh[event])).mean(axis=0)
    ax.plot(np.linspace(-300, 300, 151), data.T, label=event, linewidth=lw, color='k')
    lw += 0.4

ax.axvspan(50, 150, alpha=0.5, color='lightgrey')
ax.vlines(0, 0.5, 6, colors='black',linestyles='--')
for i in [-200, -100, 100, 200, 300]:
    ax.vlines(i, 0.5, 6, colors='black',linestyles=':', linewidth=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', ncols=6, fontsize=9, frameon=False, bbox_to_anchor=(0.6, 0.6, 0.6, 0.6))
ax.set_xlim([-310, 310])
ax.set_ylim([0, 6])

# extra plot
Brain = mne.viz.get_brain_class()
clr = 0.85
brain_kwargs = dict(alpha=1, background="white", cortex=[(clr, clr, clr), (clr, clr, clr)], size=(800, 600), views='lateral')
brain_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc')[:-1] # 69 labels
brain = Brain("fsaverage", hemi="lh", surf="pial_semi_inflated", **brain_kwargs)
brain.add_label(brain_labels[-2], hemi="lh", color="#d62728", borders=False, alpha=0.9)
brain.show_view(roll=20, azimuth=30, elevation=80, distance=400)

#### figure 1 f
fig, axs = plt.subplots(1, 1, figsize=(6, 3))
time_array = np.linspace(-300, 300, 151)
stims = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95']
color = '#ff7f0e'
lw = 0.5
for stim in stims[:-1]:
    axs.plot(time_array, abs(grand_ev_dict[stim].get_data(picks='EOG002')[0] * 1e6),
            linewidth=lw, label=stim, color=color)
    lw += 0.4
for stim in stims[-1:]:
    axs.plot(time_array, abs(grand_ev_dict[stim].get_data(picks='EOG002')[0] * 1e6),
            linewidth=lw, label=stim, color='k')
    
axs.axvspan(50, 180, alpha=0.4, color='lightgrey')
axs.legend(fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.1, 0.6, 0.6))
axs.spines['top'].set_visible(False); axs.spines['right'].set_visible(False)
for i in [-200, -100, 100, 200, 300]:
    axs.vlines(i, -10, 60, colors='black',linestyles=':', linewidth=0.5)
axs.vlines(0, -10, 60, colors='black',linestyles='--')
axs.set_ylabel(f'EOG amplitude at 70 dB (µv)')
axs.set_xlabel(f'Time (ms)')
axs.set_ylim([-10, 60])

#### figure 2 ####
# plotting Figure 2 a
data1 = grand_ev_dict['GO_60'].get_data(picks='EOG002')[0] * 1e6
data2 = grand_ev_dict['GO_70'].get_data(picks='EOG002')[0] * 1e6

fig, ax = plt.subplots(1, 1, figsize=(6,3))
ax.plot(np.linspace(-300, 300, 151), abs(data1), label='GO_60', color='#ff7f0e', linewidth=2)
ax.plot(np.linspace(-300, 300, 151), abs(data2), label='GO_70', color='#9467bd', linewidth=2)
ax.vlines(0, -10, 60, colors='black',linestyles='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_ylim([-12, 62])
ax.set_xlim([-310, 310])
for i in [-200, -100, 100, 200, 300]:
    ax.vlines(i, -10, 60, colors='black',linestyles=':', linewidth=0.5)

# plotting Figure 2 b
directory = '/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg1_morphed'
stc_files_list_rh = {'GO_60': [], 'GO_70': []}
stc_files_list_lh = {'GO_60': [], 'GO_70': []}
events = ['GO_60', 'GO_70']

brain_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc')[:-1] # 68 labels
fs_dir = fetch_fsaverage(verbose=False)
src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(src_fname, verbose=False)

for event in events:
    for filename in sorted(os.listdir(directory)): 
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and f.endswith("-lh.stc") and event in f and '697' not in f and '750' not in f and '853' not in f:
                stc = mne.read_source_estimate(fname=f, subject='fsaverage')
                rh_data = stc.extract_label_time_course(brain_labels[-1], src, mode='mean', verbose=False)
                lh_data = stc.extract_label_time_course(brain_labels[-2], src, mode='mean', verbose=False)
                stc_files_list_rh[event].append(rh_data)
                stc_files_list_lh[event].append(lh_data)

fig, ax = plt.subplots(1, 1, figsize=(6,3))
colors = ['#d62728', '#1f77b4'] 
for event, clr in zip(events, colors):
    data = np.squeeze(np.array(stc_files_list_rh[event])).mean(axis=0)
    # data_std = np.squeeze(np.array(stc_files_list_rh[event])).std(axis=0)
    data_std = stats.sem(a=np.squeeze(np.array(stc_files_list_rh[event])), axis=0)
    ax.plot(np.linspace(-300, 300, 151), data.T, label=event, color=clr)
    ax.fill_between(np.linspace(-300, 300, 151), data.T - data_std.T,
                    data.T + data_std.T, color=clr, alpha=0.1, edgecolor="none")
ax.vlines(0, 0.5, 8, colors='black',linestyles='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=9, frameon=False)
ax.set_xlim([-310, 310])
for i in [-200, -100, 100, 200, 300]:
    ax.vlines(i, 0.5, 8, colors='black',linestyles=':', linewidth=0.5)

# plotting Figure 2 c
fig, ax = plt.subplots(1, 1, figsize=(4,4))
# palette_color = 'Set1'
palette_color = ['#1f77b4', '#d62728'] 
order_1 = ['GO_60', 'GO_70']
order_2 = ['GO_70']
df1 = df[df['Hemisphere']=='rh']
sns.boxplot(data=df, x='Stimulus', y='peak value (30ms)', fill=False, linewidth=2, hue='Hemisphere', legend=False,
            saturation=0.6, palette=palette_color, order=order_1, gap=0.2, ax=ax)
sns.stripplot(data=df, x='Stimulus', y='peak value (30ms)', hue='Hemisphere', legend=False,
            dodge=True, size=4, palette=palette_color, order=order_1, ax=ax)
ax.set_ylim([0, 13])
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# plotting Figure 2 e
grand_ev_dict['GO_60'].plot_topomap(times=0.1, time_unit='ms', contours=6, vlim=(-240, 240))

# plotting Figure 2 d
brain_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc')[:-1] # 68 labels
fig, ax = plt.subplots(1,1, figsize=(10, 4))
colors = np.array(sns.color_palette('Set1'))[[1, 2, 3, 4, 6, 0, 8],:]

# data = [17, 1, 1, 1, 2, 1]
data = [13, 2, 3, 1, 2, 1, 1]

# ingredients = ['transverstemporal-rh', 'bankssts-rh', 'entorhinal-rh', 'temporalpole-rh']
ingredients = [brain_labels[67].name, brain_labels[1].name,
                brain_labels[9].name, brain_labels[0].name, brain_labels[66].name,
                brain_labels[32].name, brain_labels[42].name]
def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%"
wedges, autotexts = ax.pie(data, textprops=dict(color="w"), colors=colors)
legend_names = [f'{name} ({round(number/23*100, 1)}%)' for name, number in zip(ingredients, data)]
ax.legend(wedges, legend_names, loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=12)


#### figure 3 ####

# plotting Figure 3 a
order_1 = ['GP70_i0', 'GP70_i60', 'GP70_i120', 'GP70_i240']
order_2 = ['GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240']
mask = df['Stimulus'].isin(order_1)
df1 = df[mask]
df2 = df1[df1['Hemisphere']=='rh']
fig, ax = plt.subplots(1, 1, figsize=(5,3))
palette_color = ['grey']
sns.boxplot(data=df2, x='Stimulus', y='EOG area inhibition', width=0.5, fill=False, linewidth=2,
            saturation=0.75, palette=palette_color, order=order_1, ax=ax)
sns.stripplot(data=df2, x='Stimulus', y='EOG area inhibition',
            dodge=False, size=3, palette=palette_color, order=order_1, ax=ax, legend=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1))
ax.set_ylim([-50, 100])
ax.hlines(0, -0.6, 3.6, colors='black',linestyles='--')
ax.hlines(50, -0.6, 3.6, colors='grey',linestyles=':')
ax.set_yticks([-50, 0, 50, 100])

# plotting Figure 3 b
order_1 = ['GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240']
order_2 = ['GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240']
mask = df['Stimulus'].isin(order_2)
df1 = df[mask]
fig, ax = plt.subplots(1, 1, figsize=(9,3))
palette_color = ['#1f77b4', '#d62728']
sns.boxplot(data=df1, x='Stimulus', y='area inhibition (30ms)', hue='Hemisphere', width=0.8, fill=False, gap=.1, linewidth=2,
            saturation=0.75, palette=palette_color, order=order_2, ax=ax)
sns.stripplot(data=df1, x='Stimulus', y='area inhibition (30ms)', hue='Hemisphere',
            dodge=True, size=3, palette=palette_color, order=order_2, ax=ax, legend=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.1, 1))
ax.set_ylim([-50, 100])
ax.hlines(0, -0.6, 3.6, colors='black',linestyles='--')
ax.hlines(50, -0.6, 3.6, colors='grey',linestyles=':')
ax.set_yticks([-50, 0, 50, 100])