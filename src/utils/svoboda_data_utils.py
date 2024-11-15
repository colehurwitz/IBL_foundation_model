import pandas as pd
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt

'''
functions for preparing spike trains 
for lfads or other analyses
'''

def get_trial_spikes_and_licks(hit_left_trials, hit_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end, if_opto = False, opto_start = None, opto_type = None):
    '''
    get spike times from units and trials
    '''
    
    before = 0.5
    after = 3.5

    trial_spikes = {id:{'left': [], 'right': []} for id in ids_sort_by_area}
    licks = {'left': {'left_lick':[], 'right_lick':[]}, 'right':  {'left_lick':[], 'right_lick':[]}}
    behavior_data = {'left':{'ts':[], 'jaw': [], 'nose': [], 'tongue': []}, 'right': {'ts':[], 'jaw': [], 'nose': [], 'tongue': []}}

    opto_type_sorted = []

    for trials_of_interest, trial_type in zip([hit_left_trials, hit_right_trials], ['left', 'right']):
        for trial_id in trials_of_interest.trial_uid:
            trial = trials_of_interest.loc[trials_of_interest.trial_uid == trial_id]
            trial_start = trial.start_time.values[0]
            trial_end = trial.stop_time.values[0]
            
            #get the sample start time
            sample_start_in_trial = sample_start[(sample_start>trial_start) & (sample_start<trial_end)]

            delay_start_in_trial = delay_start[(delay_start>trial_start) & (delay_start<trial_end)] 
            delay_end_in_trial = delay_end[(delay_end>trial_start) & (delay_end<trial_end)]

            delay_duration = delay_end_in_trial[0] - delay_start_in_trial[0]
            
            if (delay_duration>1.1) & (delay_duration<1.3):
                
                aligned_left_lick = left_lick - sample_start_in_trial
                aligned_right_lick = right_lick - sample_start_in_trial
                
                aligned_left_lick = aligned_left_lick[(-before < aligned_left_lick) & (aligned_left_lick < after)]
                aligned_right_lick = aligned_right_lick[(-before < aligned_right_lick) & (aligned_right_lick < after)]

                licks[trial_type]['left_lick'].append(aligned_left_lick)
                licks[trial_type]['right_lick'].append(aligned_right_lick)

                # get aligned behavior data
                aligned_ts = nose.timestamps - sample_start_in_trial

                ts = nose.timestamps[(-before < aligned_ts) & (aligned_ts < after)]

                ref_time = np.arange(-before, after, 0.01)
                matched_ind = np.searchsorted(ref_time, aligned_ts[(-before < aligned_ts) & (aligned_ts < after)])
                matched_ind -= 1 #to handle the index out of bound error

                insert_ind, ts_ind = np.unique(matched_ind, return_index=True)

                tmp = np.zeros_like(ref_time)
                tmp[insert_ind] = ts[ts_ind]
                behavior_data[trial_type]['ts'].append(tmp)

                aligned_nose = nose.data[(-before < aligned_ts) & (aligned_ts < after),:]
                tmp = np.zeros((len(ref_time), 3))
                tmp[insert_ind] = aligned_nose[ts_ind]
                behavior_data[trial_type]['nose'].append(tmp)

                aligned_tongue = tongue.data[(-before < aligned_ts) & (aligned_ts < after),:]
                tmp = np.zeros((len(ref_time), 3))
                tmp[insert_ind] = aligned_tongue[ts_ind]
                behavior_data[trial_type]['tongue'].append(tmp)

                aligned_jaw = jaw.data[(-before < aligned_ts) & (aligned_ts < after),:]
                tmp = np.zeros((len(ref_time), 3))
                tmp[insert_ind] = aligned_jaw[ts_ind]
                behavior_data[trial_type]['jaw'].append(tmp)

                if if_opto:
                    opto_type_sorted.append(opto_type[(opto_start>trial_start) & (opto_start<trial_end)])
                    
                for id in ids_sort_by_area:
                    unit_spike_times = units[units.id==id]['spike_times'].values[0]
                    
                    # Compute spike times relative to stimulus onset
                    aligned_spikes = unit_spike_times - sample_start_in_trial
                    # Keep only spike times in a given time window around the stimulus onset
                    aligned_spikes = aligned_spikes[
                        (-before < aligned_spikes) & (aligned_spikes < after)
                    ]
                                
                    trial_spikes[id][trial_type].append(aligned_spikes)

    #turn list of list into array for items in behavior_data
    for key in behavior_data.keys():
        for key2 in behavior_data[key].keys():
            behavior_data[key][key2] = np.array(behavior_data[key][key2])

    if if_opto:
        return trial_spikes, licks, behavior_data, opto_type_sorted 
    else:
        return trial_spikes, licks, behavior_data


def time_to_sequence(spike_time_trial, T, timestep, before=0.5):
    '''
    from spike time to spike train for 1 trial
    '''
    bins = np.arange(0, T*timestep + timestep, timestep)
    f, _ = np.histogram(np.array(spike_time_trial)+before, bins=bins)
    return f


def get_spike_data(spike_train_dict, ids_sort_by_area, before=0.5, after=3.5, timestep=0.01):
    '''
    from spike times to spike trains for neuron population
    '''
    T = int((after+before)/timestep)
    n = len(ids_sort_by_area)

    spike_data_combine_all = {}
    n_trial_all = {}
    for type, trial_spikes_OI in spike_train_dict.items():
        
        spike_data_all={}
        for trial_type in ['left', 'right']:
            n_trial = len(trial_spikes_OI[ids_sort_by_area.loc[0]][trial_type])
            spike_data = np.zeros((n, T, n_trial))
            #for ind, id in enumerate(ids.values[:,0]):
            for ind, id in enumerate(ids_sort_by_area):
            
                spike_time = trial_spikes_OI[id][trial_type]
            
                for trial_i, spike_time_trial in enumerate(spike_time):
                    spike_data[ind, :, trial_i] = time_to_sequence(spike_time_trial, T, timestep)
                    
            spike_data_all[trial_type] = spike_data
        
        spike_data_left = spike_data_all['left']
        spike_data_right = spike_data_all['right']
        
        spike_data_combine = np.concatenate((spike_data_left, spike_data_right), axis=2)
        spike_data_combine_all[type] = spike_data_combine

        n_trial_all[type] = {'left': spike_data_left.shape[2], 'right': spike_data_right.shape[2]}

    return spike_data_combine_all, n_trial_all



def main_get_spike_trains(nwbfile):
    '''
    main function to get spike trains of various trial conditions from nwbfile
    '''

    # get trial info and unit info
    trials_df = nwbfile.trials.to_dataframe()
    units_df = nwbfile.units.to_dataframe()

    # get behavior data
    jaw = nwbfile.acquisition['BehavioralTimeSeries']['Camera0_side_JawTracking']
    nose = nwbfile.acquisition['BehavioralTimeSeries']['Camera0_side_NoseTracking']
    tongue = nwbfile.acquisition['BehavioralTimeSeries']['Camera0_side_TongueTracking']

    ################
    hit_trials = trials_df.query("photostim_onset == 'N/A' and outcome == 'hit' and early_lick == 'no early' and auto_water==0 and free_water==0")
    miss_trials = trials_df.query("photostim_onset == 'N/A' and outcome == 'miss' and early_lick == 'no early'and auto_water==0 and free_water==0")
    ignore_trials = trials_df.query("photostim_onset == 'N/A' and outcome == 'ignore' and early_lick == 'no early'and auto_water==0 and free_water==0")

    hit_left_trials = hit_trials.query("trial_instruction == 'left'")
    hit_right_trials = hit_trials.query("trial_instruction == 'right'")

    miss_left_trials = miss_trials.query("trial_instruction == 'left'")
    miss_right_trials = miss_trials.query("trial_instruction == 'right'")

    ignore_left_trials = ignore_trials.query("trial_instruction == 'left'")
    ignore_right_trials = ignore_trials.query("trial_instruction == 'right'")

    opto_hit_trials = trials_df.query("photostim_onset != 'N/A' and outcome == 'hit' and early_lick == 'no early' and auto_water==0 and free_water==0")
    opto_miss_trials = trials_df.query("photostim_onset != 'N/A' and outcome == 'miss' and early_lick == 'no early'and auto_water==0 and free_water==0")
    opto_ignore_trials = trials_df.query("photostim_onset != 'N/A' and outcome == 'ignore' and early_lick == 'no early'and auto_water==0 and free_water==0")

    opto_hit_left_trials = opto_hit_trials.query("trial_instruction == 'left'")
    opto_hit_right_trials = opto_hit_trials.query("trial_instruction == 'right'")

    opto_miss_left_trials = opto_miss_trials.query("trial_instruction == 'left'")
    opto_miss_right_trials = opto_miss_trials.query("trial_instruction == 'right'")

    opto_ignore_left_trials = opto_ignore_trials.query("trial_instruction == 'left'")
    opto_ignore_right_trials = opto_ignore_trials.query("trial_instruction == 'right'")
    #############

    units = units_df.query("classification == 'good'").reset_index()
    ids_sort_by_area = units.sort_values(by='anno_name').id

    sorted_area = units.sort_values(by='anno_name').anno_name.unique()
    area_value_dict = {val: i for i, val in enumerate(sorted_area)}
    area_value = [area_value_dict[area] for area in units.sort_values(by='anno_name').anno_name]


    sample_start = nwbfile.acquisition['BehavioralEvents']['sample_start_times'].timestamps[:]
    delay_start = nwbfile.acquisition['BehavioralEvents']['delay_start_times'].timestamps[:]
    delay_end = nwbfile.acquisition['BehavioralEvents']['delay_stop_times'].timestamps[:]

    left_lick = nwbfile.acquisition['BehavioralEvents']['left_lick_times'].timestamps[:]
    right_lick = nwbfile.acquisition['BehavioralEvents']['right_lick_times'].timestamps[:]

    # get behavior data

    trial_spikes_hit, licks_hit, behavior_hit = get_trial_spikes_and_licks(hit_left_trials, hit_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end)
    trial_spikes_miss, licks_miss, behavior_miss = get_trial_spikes_and_licks(miss_left_trials, miss_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end)
    trial_spikes_ignore, licks_ignore, behavior_ignore = get_trial_spikes_and_licks(ignore_left_trials, ignore_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end)

    opto_start = nwbfile.acquisition['BehavioralEvents']['photostim_start_times'].timestamps[:]
    opto_type  = nwbfile.acquisition['BehavioralEvents']['photostim_start_times'].control[:]

    trial_spikes_opto_hit, licks_opto_hit, behavior_opto_hit, opto_type_sorted_hit = get_trial_spikes_and_licks(opto_hit_left_trials, opto_hit_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end, if_opto = True, opto_start = opto_start, opto_type = opto_type) 
    trial_spike_opto_miss, licks_opto_miss, behavior_opto_miss, opto_type_sorted_miss = get_trial_spikes_and_licks(opto_miss_left_trials, opto_miss_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end, if_opto = True, opto_start = opto_start, opto_type = opto_type)
    trial_spike_opto_ignore, licks_opto_ignore, behavior_opto_ignore, opto_type_sorted_ignore = get_trial_spikes_and_licks(opto_ignore_left_trials, opto_ignore_right_trials, left_lick, right_lick, jaw, nose, tongue, units, ids_sort_by_area, sample_start, delay_start, delay_end, if_opto = True, opto_start = opto_start, opto_type = opto_type)
    
    spike_trains_dict = {'hit': trial_spikes_hit, 'miss': trial_spikes_miss, 'ignore': trial_spikes_ignore, 
                         'opto_hit': trial_spikes_opto_hit, 'opto_miss': trial_spike_opto_miss, 'opto_ignore': trial_spike_opto_ignore}

    licks_dict = {'hit': licks_hit, 'miss': licks_miss, 'ignore': licks_ignore,
                    'opto_hit': licks_opto_hit, 'opto_miss': licks_opto_miss, 'opto_ignore': licks_opto_ignore}
    
    opto_type_sorted_dict={'hit': opto_type_sorted_hit, 'miss': opto_type_sorted_miss, 'ignore': opto_type_sorted_ignore}

    behavior_dict = {'hit': behavior_hit, 'miss': behavior_miss, 'ignore': behavior_ignore, 
                     'opto_hit': behavior_opto_hit, 'opto_miss': behavior_opto_miss, 'opto_ignore': behavior_opto_ignore}






    spike_data_combine_all, n_trial_all = get_spike_data(spike_trains_dict, ids_sort_by_area)

    return spike_data_combine_all, n_trial_all, licks_dict, behavior_dict, units, ids_sort_by_area, opto_type_sorted_dict, sorted_area, area_value, area_value_dict