# Support functions for EMG analysis of MEP recordings, induced by TMS stimulations to the M1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal


def plot_pre_post_filtering(original_df, filtered_df):
    '''Plot two df side by side. Useful when comparing raw and filtered signals'''

    y_min = (min(np.min(original_df.values), np.min(filtered_df.values)))*1.15 #should be a negative value
    y_max = (max(np.max(original_df.values), np.max(filtered_df.values)))*1.15

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6), dpi=300)
    sns.lineplot(data=original_df, ax=ax1, legend=False)
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('mV')
    ax1.set_ylim([y_min,y_max])

    sns.lineplot(data=filtered_df, ax=ax2, legend=False)
    ax2.set_title('Signal after mean reduction')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('mV')
    ax2.set_ylim([y_min,y_max])



def find_tms_pulse(emg_signal):
    '''Finds the location of the TMS pluse artifact by looking at signal derivatives
    Input: EMG recordings of a frame containing a single TMS pulse
    Output: return the number of row (i.e. time) in which the TMS pulse was given '''
    
    peaks, _ = scipy.signal.find_peaks(x = emg_signal, prominence=0.05)
    if len(peaks)==0:
        #try looking for a trough (peak of the negative signal)
        peaks, _ = scipy.signal.find_peaks(x = -emg_signal, prominence=0.05)
    
    return peaks[0]



def is_mep(emg_signal):
    '''Checks if there was a valid motor evoked potential (MEP) in the signal
    Input: EMG recordings of a single TMS pulse
    Output: True if there was a valid MEP, False otherwise'''
    
    peaks, _ = scipy.signal.find_peaks(x = emg_signal, prominence=0.05)
    if len(peaks)>1:
        return True
    else:
        return False



def find_mep_timing(emg_signal):
    ''' Returns the exact timing of the peak MEP in ms. If no MEP was detected, prints an error msg
    Input: EMG recordings of a single TMS pulse'''

    if not is_mep(emg_signal):
        print('No MEP detected')
        return []
    else:
        peaks, _ = scipy.signal.find_peaks(x = emg_signal, prominence=0.05)
        return peaks[1]



def get_mep_size(emg_signal):
    '''Returns the size of the peak-to-peak amplitude of the MEP signal (in mV)
    Input: EMG recordings of a single TMS pulse'''

    peak_emg = find_mep_timing(emg_signal)
    window = [peak_emg-50, peak_emg+50]
    
    mep_size = np.max(emg_signal[window[0]:window[1]]) - np.min(emg_signal[window[0]:window[1]])
    return mep_size



def draw_detected(emg_signal):
    '''Plot the recorded EMG signal with markings over it's TMS pulse and MEP'''
    
    peak_emg = find_mep_timing(emg_signal)
    if peak_emg == []:
        print('No MEP detected')
    else:
        window = [peak_emg-50, peak_emg+50]
        plt.figure(figsize=(14,4))
        sns.lineplot(y= emg_signal, x=emg_signal.index, legend=False, color='black')
        
        #draw a shaded area where we expect the MEP to occur
        y_min = np.min(emg_signal)*1.1
        y_max = np.max(emg_signal)*1.1
        plt.fill_between(x=emg_signal.index, y1=y_min, y2=y_max, where=(emg_signal.index >= window[0]) & (emg_signal.index <= window[1]), color='indigo', alpha=0.2)
        
        #draw a vertical line where we found the TMS pulse
        tms_pulse = find_tms_pulse(emg_signal)
        plt.axvline(x=tms_pulse, color='indigo', linestyle='solid', linewidth=0.9)
        plt.xlabel('Time (ms)')
        plt.ylabel('mV')



def plot_recruitment_curve(df):
    '''Plot the entire recruitment curve. Assuming an experimental design of
    12 pulses at 100% RMT, then 6 at each of the following intensities: 110%, 120%, 130%, 140%, 150%
    Input: full data frame, with time as rows, and trial as columns
    Output: a list of the mep sizes'''

    recruitment_curve = [None]*6
    mep_sizes = []
    for col, _ in df.iteritems():
        if is_mep(df[col]):
            mep_sizes.append(get_mep_size(df[col]))
        else:
            mep_sizes.append(None)
    
    sum_meps = 0; counter = 0
    for i in range(12):
        if mep_sizes[i] != None:
            sum_meps += mep_sizes[i]
            counter += 1
    recruitment_curve[0] = sum_meps/counter

    sum_meps = 0; counter = 0
    for tms_intensity in range(1,6):
        #Stimulation at 100% RMT is given 12 times. Then, each stimulation intensity was given 6 times
        location = 6+tms_intensity*6
        for i in range(6):
            if mep_sizes[location+i] != None:
                sum_meps += mep_sizes[location+i]
                counter += 1
        recruitment_curve[tms_intensity] = sum_meps/counter

    #set relative stimulation intensities to be the x value of the recruitment curve plot
    stimulation_intensities = [100]*12 + [110]*6+ [120]*6+ [130]*6+ [140]*6+ [150]*6

    rec_curve_long_format = {'Stimulation Intensities':stimulation_intensities, 'MEP size':mep_sizes}
    
    sns.lineplot(data=rec_curve_long_format, x='Stimulation Intensities', y='MEP size', color='indigo', markers=True, linewidth=1.5)
    plt.xlabel('TMS Intensity (%RMT)')
    return recruitment_curve
