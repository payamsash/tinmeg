import mne
import numpy as np
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse
from tqdm.notebook import tqdm
import os


#### Source localizing ####

# Cortical surface reconstruction (+bem, +head_model) with FreeSurfer (as an example for one subject)
# $ export FREESURFER_HOME=/Applications/freesurfer/7.4.1
# $ export SUBJECTS_DIR=$FREESURFER_HOME/subjects
# $ source $FREESURFER_HOME/SetUpFreeSurfer.sh
# $ recon-all -s 0863 -i /Users/payamsadeghishabestari/KI_MEG/MRI/0863/00000003/00000001.dcm 
# $ recon-all -all -subjid 0863

# setting up watershed BEM files 
subject = '0863'
subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
mne.bem.make_watershed_bem(subject, subjects_dir=None,
                            overwrite=False, volume='T1', atlas=False,
                            gcaatlas=False, preflood=None, show=False,
                            copy=True, T1=None, brainmask='ws.mgz', verbose=None)



#### Concatenating, creating evoked objects and grand averaging ####
# Create epochs dictionary (some needs concatenating)
epochs_folder = '/Users/payamsadeghishabestari/KI_MEG/epochs/tinmeg1'
epochs_file = {}
for f in sorted(os.listdir(epochs_folder)):
    file = os.path.join(epochs_folder, f)
    if file.endswith("-epo.fif") and '697' not in file and '750' not in file and '853' not in file and '841' not in file:
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


#### looping over subjects ####

po_stims = ['PO60_70', 'PO60_75', 'PO60_80', 'PO60_85', 'PO60_90', 'PO60_95'] # for tinmeg1
subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
method = "dSPM"
snr = 3.0
lambda2 = 1.0 / snr**2

for subject in np.array(list(epochs_file.keys()))[[11,12]]: # list(epochs_file.keys())
    
    subject_idx = list(epochs_file.keys()).index(subject)
    report = mne.Report(title=f'source_localization_report_subject_{subject}', verbose=False)
    
    # Setting up the surface source space
    print(f'Setting up bilateral hemisphere surface-based source space with subsampling for subject {subject} ...')
    src = mne.setup_source_space(f'{subject}', spacing="oct6", subjects_dir=subjects_dir, n_jobs=-1, verbose=None)

    # Setting up the boundary-element model (BEM) 
    print(f'Creating a BEM model for subject ...')
    bem_model = mne.make_bem_model(subject=f'{subject}', ico=4, subjects_dir=subjects_dir, verbose=False)  
    bem = mne.make_bem_solution(bem_model, verbose=False)
    report.add_bem(subject=f'{subject}', subjects_dir=subjects_dir, title="MRI & BEM", decim=10, width=512)

    # Aligning coordinate frame (coregistration MEG-MRI)
    print(f'Coregistering MRI with a subjects head shape ...')
    # info = grnd_ev_dict['PO60_80'][subject_idx].info # tinmeg1
    info = grnd_ev_dict['PO_00'][subject_idx].info # tinmeg3
    coreg = Coregistration(info, f'{subject}', subjects_dir, fiducials='auto')
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=40, nasion_weight=2.0, verbose=False) # refining with ICP
    coreg.omit_head_shape_points(distance=5.0 / 1000) # omitting bad points (larger than 5mm)
    coreg.fit_icp(n_iterations=40, nasion_weight=10, verbose=False) # final fitting
    fname_trans = f'/Users/payamsadeghishabestari/KI_MEG/trans/{subject}-trans.fif'
    mne.write_trans(fname_trans, coreg.trans, overwrite=True, verbose=False)
    report.add_trans(trans=fname_trans, info=info, subject=f'{subject}',
                    subjects_dir=subjects_dir, alpha=1.0, title="Co-registration")

    # Computing the forward solution
    print(f'Computing the forward solution ...')
    fwd = mne.make_forward_solution(info, trans=coreg.trans, src=src, bem=bem, meg=True,
                                    eeg=False, mindist=5.0, n_jobs=None, verbose=False)

    # Computing the regularized noise-covariance matrix (consider the notes)
    print(f'Estimate the noise covariance of the recording ...')
    epochs = mne.read_epochs(fname=epochs_file[subject], verbose=False)
    noise_cov = mne.compute_covariance(epochs[po_stims], tmax=0.0, method=("empirical", "shrunk"),
                                        verbose=False) # using the epochs baseline 
    
    # Computing the minimum-norm inverse solution
    print(f'Computing the minimum-norm inverse solution ...')
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)

    # Compute source estimate object
    print(f'Computing and saving the source estimate object ...')
    for key_id in list(grnd_ev_dict.keys()):
        stc = apply_inverse(grnd_ev_dict[key_id][subject_idx], inverse_operator, lambda2, method=method, pick_ori=None,
                            return_residual=False, verbose=False)
        fname_stc = f'/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg3/{subject}_{key_id}'
        stc.save(fname=fname_stc, overwrite=True, verbose=False)

    # saving report
    fname_report = f'/Users/payamsadeghishabestari/KI_MEG/reports/source_localization_report_subject_{subject}.html'
    report.save(fname=fname_report, open_browser=False, overwrite=True, verbose=False)


#### Morphing to freesurfer template brain ####

subjects_dir = '/Applications/freesurfer/7.4.1/subjects'
fname_fsaverage_src = '/Users/payamsadeghishabestari/mne_data/MNE-fsaverage-data/fsaverage/bem/fsaverage-ico-5-src.fif'
directory = '/Users/payamsadeghishabestari/KI_MEG/stcs/tinmeg3' 
src_to = mne.read_source_spaces(fname_fsaverage_src)

# iterate over files in that directory
stc_files_list = []
for filename in sorted(os.listdir(directory)): 
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith("-lh.stc"): # or -rh
        stc_files_list.append(f)

# morphing from oct-6 to ico-5 at fsaverage
for stc_file in tqdm(stc_files_list):
    stc = mne.read_source_estimate(fname=stc_file, subject=f'{stc_file[50:54]}')
    
    morph = mne.compute_source_morph(stc, subject_from=f'{stc_file[50:54]}',
                                    subject_to="fsaverage", subjects_dir=subjects_dir,
                                    src_to=src_to)
    stc_morph = morph.apply(stc)
    fname_stc_morph = ''.join([stc_file[:49], '_morphed', stc_file[49:]])
    stc_morph.save(fname=fname_stc_morph, overwrite=True, verbose=False)