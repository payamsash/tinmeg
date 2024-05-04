import mne
import numpy as np
import pandas as pd
import os
import os.path as op
from tqdm.notebook import tqdm
from mne.datasets import fetch_fsaverage
from scipy import stats as stats



#### making big dataframe for transverstemporal rh/lh ####
my_dict = {'subject ID': [], 'Stimulus': [], 'Hemisphere': [], 'latency': [],
        'PSNR': [], 'peak value (30ms)': [], 'peak value (100ms)': [], 'area value (30ms)': [],
        'area value (100ms)': [], 'amplitude inhibition (30ms)': [], 'area inhibition (30ms)': [],
        'amplitude inhibition (100ms)': [], 'area inhibition (100ms)': [], 'EOG peak': [],
        'EOG ptp': [], 'EOG area (30ms)': [], 'EOG area (100ms)': [], 'EOG peak latency': []} 

events = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95',
        'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95',
        'GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240',
        'GP70_i0', 'GP70_i60', 'GP70_i120', 'GP70_i240',
        'GO_60', 'GO_70']

# events = ['GPP_00', 'GPG_00', 'PO_00', 'GO_00', 'PPP_00', 'PPG_00',
#         'GPP_03', 'GPG_03', 'PO_03', 'GO_03',
#         'GPP_08', 'GPG_08', 'PO_08', 'GO_08',
#         'GPP_30', 'GPG_30', 'PO_30', 'GO_30',
#         'GPP_33', 'GPG_33', 'PO_33', 'GO_33',
#         'GPP_38', 'GPG_38', 'PO_38', 'GO_38',
#         'GPP_80', 'GPG_80', 'PO_80', 'GO_80',
#         'GPP_83', 'GPG_83', 'PO_83', 'GO_83',
#         'GPP_88', 'GPG_88', 'PO_88', 'GO_88']

# events = ['GPP_00', 'GPG_00', 'PO_00', 'GO_00', 'PPP_00', 'PPG_00',
#         'GPP_03', 'GPG_03', 'PO_03', 'GO_03',
#         'GPP_08', 'GPG_08', 'PO_08', 'GO_08']

subjects = ['539', '697', '750', '756', '832', '835', '836', '838', '839',
            '840', '841', '842', '844', '845', '847', '849', '850', '852', '853',
            '856', '857', '858', '859', '861', '862', '863'] # should I exclude 3 subjects?

# subjects = ['916', '979', '980', '981', '982', '983', '984', '986', '988']

# subjects = ['1004', '1006', '1008', '1009', '1017', '1021', '1025', '1031',
#             '1032', '1033', '1034', '1035', '1037', '1038', '1044', '1045', '1047', '1048']

brain_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc')[:-1] # 68 labels
fs_dir = fetch_fsaverage(verbose=False)
src_fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(src_fname, verbose=False)
directory_stcs = '/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg2_morphed'
directory_eps = '/Users/payamsadeghishabestari/KI_MEG/epochs/tinmeg2/epochs_bads_dropped'

# looping over all files
for subject in tqdm(subjects):
    for stim in events:
        for hemi in ['lh', 'rh']:
            for filename in sorted(os.listdir(directory_stcs)): 
                f = os.path.join(directory_stcs, filename)
                if os.path.isfile(f) and f.endswith(f"-{hemi}.stc") and stim in f and subject in f:
                    
                    # reading source estimate file
                    stc_fname = f
                    stc = mne.read_source_estimate(fname=stc_fname, subject='fsaverage')
                    
                    # reading epoch file for eog response
                    for filename in sorted(os.listdir(directory_eps)): 
                        f = os.path.join(directory_eps, filename)
                        if os.path.isfile(f) and f.endswith('-epo.fif') and subject in f:
                            ep_fname = f
                            ep = mne.read_epochs(fname=ep_fname, preload=True, verbose=False)
                    
                    eog_peak_value = abs(np.squeeze(ep[stim].get_data(picks=['EOG002'])).mean(axis=0)[75:].min() * 1e6) # only positive
                    argmin = np.squeeze(ep[stim].get_data(picks=['EOG002'])).mean(axis=0)[75:].argmin() + 75
                    eog_peak_time = np.linspace(-300, 300, 151)[argmin]
                    (t1, t2) = (argmin - 5, argmin + 4)
                    (t3, t4) = (argmin - 14, argmin + 13)
                    my_dict['EOG peak'].append(eog_peak_value)
                    my_dict['EOG ptp'].append(np.ptp(np.squeeze(ep[stim].get_data(picks=['EOG002'])).mean(axis=0)[75:]) * 1e6)
                    my_dict['EOG area (30ms)'].append(abs(np.squeeze(ep[stim].get_data(picks=['EOG002'])).mean(axis=0)[t1:t2].sum() * 1e6))
                    my_dict['EOG area (100ms)'].append(abs(np.squeeze(ep[stim].get_data(picks=['EOG002'])).mean(axis=0)[t3:t4].sum() * 1e6))
                    my_dict['EOG peak latency'].append(eog_peak_time)
                    
                    # localizing the brain label
                    if hemi == 'rh':
                        bl_idx = -1 # transversetemporal-rh
                    if hemi == 'lh':
                        bl_idx = -2 # transversetemporal-lh

                    # computing some params in stc object   
                    tcs_noise_avg = stc.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,:76].mean()
                    tcs_peak_100ms = stc.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,87:114].max()
                    tcs_peak_30ms = stc.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,96:105].max()
                    tcs_area_100ms = stc.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,87:114].sum()
                    tcs_area_30ms = stc.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,96:105].sum()
                    
                    # computing inhibition indexes for tinmeg1
                    if '60' in stim and '70' not in stim:
                        stc_fname_stn = f'/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg1_morphed/{subject}_PO60_90-lh.stc-lh.stc'
                        stc_stn = mne.read_source_estimate(fname=stc_fname_stn, subject='fsaverage')
                        tcs_peak_30ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx],
                                                                                src, mode='mean', verbose=False)[:,96:105].max()
                        tcs_area_30ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,96:105].sum()
                        tcs_peak_100ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx],
                                                                                src, mode='mean', verbose=False)[:,87:114].max()
                        tcs_area_100ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,87:114].sum()

                    if '70' in stim:
                        stc_fname_stn = f'/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg1_morphed/{subject}_PO70_90-lh.stc-lh.stc'
                        stc_stn = mne.read_source_estimate(fname=stc_fname_stn, subject='fsaverage')
                        tcs_peak_30ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx],
                                                                                src, mode='mean', verbose=False)[:,96:105].max()
                        tcs_area_30ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,96:105].sum()
                        tcs_peak_100ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx],
                                                                                src, mode='mean', verbose=False)[:,87:114].max()
                        tcs_area_100ms_stn = stc_stn.extract_label_time_course(brain_labels[bl_idx], src, mode='mean', verbose=False)[:,87:114].sum()
            
                    
                    my_dict['amplitude inhibition (30ms)'].append((1 - (tcs_peak_30ms / tcs_peak_30ms_stn)) * 100)
                    my_dict['area inhibition (30ms)'].append((1 - (tcs_area_30ms / tcs_area_30ms_stn)) * 100)
                    my_dict['amplitude inhibition (100ms)'].append((1 - (tcs_peak_100ms / tcs_peak_100ms_stn)) * 100)
                    my_dict['area inhibition (100ms)'].append((1 - (tcs_area_100ms / tcs_area_100ms_stn)) * 100)

                    # putting in the dictionary
                    my_dict['subject ID'].append(subject)
                    my_dict['Stimulus'].append(stim)
                    my_dict['Hemisphere'].append(hemi)
                    my_dict['latency'].append(stc.extract_label_time_course(brain_labels[bl_idx], src,
                                                                            mode='mean', verbose=False)[:,87:114].argmax() + 87) 
                    my_dict['PSNR'].append(20 * np.log10(tcs_peak_30ms / tcs_noise_avg))
                    my_dict['peak value (30ms)'].append(tcs_peak_30ms)
                    my_dict['peak value (100ms)'].append(tcs_peak_100ms)
                    my_dict['area value (30ms)'].append(tcs_area_30ms)
                    my_dict['area value (100ms)'].append(tcs_area_100ms)
                    
df = pd.DataFrame(my_dict)
# save it
df.to_csv('/Users/payamsadeghishabestari/KI_MEG/dataframes/tinmeg2_transverstemporal.csv')