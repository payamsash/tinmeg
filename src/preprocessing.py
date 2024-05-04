import mne
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
import numpy as np
import os
import os.path as op
from tqdm.notebook import tqdm
import time

#### Loading MEG data (movement corrected) and defining events ####

# loading subject IDs (subject 697 cant be loaded and subject 859 has different dev_head_t during the two recordings)
subjects_fname = '/Users/payamsadeghishabestari/KI_MEG/sub_date.txt'
subject_ids = np.loadtxt(fname=subjects_fname, delimiter=',', skiprows=1, usecols=1)
subject_ids = [int(s_id) for s_id in subject_ids]

# find all the subject folders
directory = '/Users/payamsadeghishabestari/KI_MEG/meg_rec_tinmeg1' 
folders_list = []
for folder in sorted(os.listdir(directory)): ## iterate over folders in that directory
    f = os.path.join(directory, folder)
    if os.path.isdir(f): ## select only folders
        folders_list.append(f)

# create a dictionary of subjects with their files
files_dict = {}
for subject_id in subject_ids:
    for folder in folders_list:
        if f'{subject_id}' in folder:
            f = os.path.join(folder, sorted(os.listdir(folder))[-1])
            files_dict[f'{subject_id}'] = [f]



#### creating event dictionary ####
# tinmeg1
keys = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95',
        'PO70_75', 'PO70_80', 'PO70_85', 'PO70_90', 'PO70_95',
        'GP60_i0', 'GP60_i60', 'GP60_i120', 'GP60_i240',
        'GP70_i0', 'GP70_i60', 'GP70_i120', 'GP70_i240',
        'GO_60', 'GO_70']
values = [40968, 36872, 34824, 33800, 33288, 33032,
        36876, 34828, 33804, 33292, 33036,
        49800, 49736, 49704, 49688,
        49804, 49740, 49708, 49692,
        16386, 16390]
events_dict_tinmeg1 = {}
for key, value in zip(keys, values):
        events_dict_tinmeg1[key] = value

# tinmeg2
keys = ['GPP_00', 'GPG_00', 'PO_00', 'GO_00', 'PPP_00', 'PPG_00',
        'GPP_03', 'GPG_03', 'PO_03', 'GO_03',
        'GPP_08', 'GPG_08', 'PO_08', 'GO_08',
        'GPP_30', 'GPG_30', 'PO_30', 'GO_30',
        'GPP_33', 'GPG_33', 'PO_33', 'GO_33',
        'GPP_38', 'GPG_38', 'PO_38', 'GO_38',
        'GPP_80', 'GPG_80', 'PO_80', 'GO_80',
        'GPP_83', 'GPG_83', 'PO_83', 'GO_83',
        'GPP_88', 'GPG_88', 'PO_88', 'GO_88']
values = [1, 2, 4, 8, 16, 32,
        49, 50, 52, 56,
        33, 34, 36, 40,
        193, 194, 196, 200,
        241, 242, 244, 248,
        225, 226, 228, 232,
        129, 130, 132, 136,
        177, 178, 180, 184,
        161, 162, 164, 168]
events_dict_tinmeg2 = {}
for key, value in zip(keys, values):
        events_dict_tinmeg2[key] = value

#tinmeg3
keys = ['GPP_00', 'GPG_00', 'PO_00', 'GO_00', 'PPP_00', 'PPG_00',
        'GPP_03', 'GPG_03', 'PO_03', 'GO_03',
        'GPP_08', 'GPG_08', 'PO_08', 'GO_08']
values = [1, 2, 4, 8, 16, 32,
        49, 50, 52, 56,
        33, 34, 36, 40]
events_dict_tinmeg3 = {}
for key, value in zip(keys, values):
        events_dict_tinmeg3[key] = value

#### Maxwell Filtering and environmental noise reduction (if necessary) ####

# loading the empty room recordings before and after exp
fname_empty_before = '/Users/payamsadeghishabestari/KI_MEG/697/empty_room_before.fif'
fname_empty_after = '/Users/payamsadeghishabestari/KI_MEG/697/empty_room_after.fif'
raw_empty_before = mne.io.read_raw_fif(fname=fname_empty_before, preload=True, allow_maxshield=True, verbose=False)
raw_empty_after = mne.io.read_raw_fif(fname=fname_empty_after, preload=True, allow_maxshield=True, verbose=False)

# compute projections for empty room recordings and concatenate them
raw_empty_before.del_proj()
raw_empty_after.del_proj()
empty_room_before_projs = mne.compute_proj_raw(raw_empty_before, n_grad=2, n_mag=2, verbose=False)
empty_room_after_projs = mne.compute_proj_raw(raw_empty_after, n_grad=2, n_mag=2, verbose=False)
extended_proj = list(np.concatenate((np.array(empty_room_before_projs), np.array(empty_room_after_projs))))

# load the experiment recording
fname = '/Users/payamsadeghishabestari/KI_MEG/697/tinmeg1-1.fif'
raw = mne.io.read_raw_fif(fname=fname, preload=True, allow_maxshield=True, verbose=False)

# estimating continous head movement
chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

# find bad channels
noisy_chs, flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw, head_pos=head_pos, verbose=True)
bads = raw.info["bads"] + noisy_chs + flat_chs
raw.info["bads"] = bads

# apply movement corection and time-signal space seperation
raw_sss = mne.preprocessing.maxwell_filter(raw, head_pos=head_pos, st_fixed=True,
                                            extended_proj=extended_proj,verbose=True)




#### Preprocessing ####

sfreq = 250
(l_freq, h_freq) = (0.1, 40)
(tmin, tmax) = (-0.3, 0.3) # baseline period of 300 ms
reject_criteria = dict(grad=4000e-13, mag=4e-12, eog=250e-6)    # T/m  # T  # V
flat_criteria = dict(mag=1e-15, grad=1e-13)  # 1 fT  # 1 fT/cm  (adding flat option to eog)
subjects = list(files_dict.keys())[:]

# reading the MEG file
for subject in tqdm(subjects): 
    start_time = time.time()
    print(subject)
    print('reading the MEG file, be patient ...')
    fname = files_dict[subject][0]
    raw = mne.io.read_raw_fif(fname=fname, preload=True, allow_maxshield=True, verbose=False)
    #if raw.last_samp / raw.info['sfreq'] < 3000:
    #    raise ValueError(f'All .fif files might not be loaded for subject {subject}')
    events_orig = mne.find_events(raw, stim_channel=None, min_duration=0.005, shortest_event=1, uint_cast=True, verbose=False) # min_duration = 0

    # delay compensation for tinmeg1 data
    #delay = int((50 / 1000) * raw.info['sfreq']) # 50 ms delay 
    #po_ids = list(events_dict_tinmeg3.values())[:11] # only PO triggers
    #for row in range(len(events_orig)):
    #    if events_orig[row][2] in po_ids:
    #        events_orig[row][0] = events_orig[row][0] - delay

    # resampling and filtering the data
    print('resampling and filtering the data, be patient, will last a while ...')
    raw, events = raw.resample(sfreq=sfreq, events=events_orig, verbose=False)
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False) 

    # creating ECG and EOG evoked responses
    ecg_evoked_meg,  ecg_evoked_grad = create_ecg_epochs(raw,
                                    verbose=False).average().apply_baseline(baseline=(None, -0.2),
                                    verbose=False).plot_joint(picks=['meg', 'grad'], show=False)
    eog_evoked_meg,  eog_evoked_grad = create_eog_epochs(raw,
                                    verbose=False).average().apply_baseline(baseline=(None, -0.2),
                                    verbose=False).plot_joint(picks=['meg', 'grad'], show=False)

    # computing ICA and remove ECG, saccade and muscle artifacts (if any) and interpolating (if any)
    print('computing ICA (this might take a while) ...')
    ica = mne.preprocessing.ICA(n_components=0.95, max_iter=800, method='infomax',
                                random_state=42, fit_params=dict(extended=True)) 
    ica.fit(raw, verbose=False) 
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="ctps", measure='zscore', verbose=False)
    if len(ecg_indices) > 0:
        ecg_component = ica.plot_properties(raw, picks=ecg_indices, verbose=False, show=False)
    emg_indices, emg_scores = ica.find_bads_muscle(raw, verbose=False)
    if len(emg_indices) > 0:
        emg_component = ica.plot_properties(raw, picks=emg_indices, verbose=False, show=False)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG002') 
    if len(eog_indices) > 0:
        eog_component = ica.plot_properties(raw, picks=eog_indices, verbose=False, show=False)

    exclude_idxs = ecg_indices + emg_indices
    ica.apply(raw, exclude=exclude_idxs, verbose=False)
    raw.interpolate_bads(verbose=False)

    # event dict selection, epoching and dropping bad epochs 
    #if fname[-43] == '1':
    #    events_dict = events_dict_tinmeg1
    #if fname[-43] == '2':
    #    events_dict = events_dict_tinmeg2
    #if fname[-43] == '3':
    #    events_dict = events_dict_tinmeg3
    
    events_dict = events_dict_tinmeg1
    print('Epoching data ...')
    epochs = mne.Epochs(raw, events, event_id=events_dict, tmin=tmin, tmax=tmax, baseline=(None, 0),
                        reject=None, flat=None, preload=True, verbose=False) 
    dropped_epochs_fig = epochs.plot_drop_log(color=(0.6, 0.2, 0.4), width=0.4, show=False)

    # creating a report
    report = mne.Report(title=f'report_subject_{subject}', verbose=False)
    report.add_raw(raw=raw, title='recording after preprocessing', butterfly=False, psd=False) 
    report.add_figure(fig=ecg_evoked_meg, title='ECG evoked MEG', image_format='PNG')
    report.add_figure(fig=ecg_evoked_grad, title='ECG evoked Gradiometer', image_format='PNG')
    report.add_figure(fig=eog_evoked_meg, title='EOG evoked MEG', image_format='PNG')
    report.add_figure(fig=eog_evoked_grad, title='EOG evoked Gradiometer', image_format='PNG')
    if len(ecg_indices) > 0:
        report.add_figure(fig=ecg_component, title='ECG component', image_format='PNG')
    if len(emg_indices) > 0:
        report.add_figure(fig=emg_component, title='EMG component', image_format='PNG')
    if len(eog_indices) > 0:
        report.add_figure(fig=eog_component, title='EOG component (saccade)', image_format='PNG')    
    report.add_figure(fig=dropped_epochs_fig, title='Dropped Epochs', image_format='PNG')
    
    # saving report and epochs
    
    fname_report = f'/Users/payamsadeghishabestari/KI_MEG/pending files/{subject}/report_subject_{subject}.html'
    fname_epoch = f'/Users/payamsadeghishabestari/KI_MEG/pending files/{subject}/epochs_subject_{subject}-epo.fif'
    report.save(fname=fname_report, open_browser=False, overwrite=True, verbose=False)
    epochs.save(fname=fname_epoch, overwrite=True, verbose=False)
    print(f'elapsed time for subject {subject} was {time.time() - start_time}')
