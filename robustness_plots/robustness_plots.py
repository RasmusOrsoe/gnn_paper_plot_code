import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import os
import matplotlib as mpl
mpl.use('pdf')
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from pandas.core.algorithms import diff

def calculate_auc(data, is_retro, target):
    if is_retro:
        if target == 'track':
            prediction_key = 'L7_PIDClassifier_FullSky_ProbTrack'
        if target == 'neutrino':
            prediction_key = 'L7_MuonClassifier_FullSky_ProbNu'
    else:
        prediction_key = target + '_pred'
    
    fpr, tpr, _ = roc_curve(data[target], data[prediction_key])
    return auc(fpr,tpr)  

def calculate_xyz_difference(data,is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    diff = np.sqrt((data['position_x'] - data['position_x%s'%post_fix])**2 + (data['position_y'] - data['position_y%s'%post_fix])**2 + (data['position_z'] - data['position_z%s'%post_fix])**2)
    return diff

def convert_to_unit_vectors(data, post_fix):
    
    data['x'] = np.cos(data['azimuth'])*np.sin(data['zenith'])
    data['y'] = np.sin(data['azimuth'])*np.sin(data['zenith'])
    data['z'] = np.cos(data['zenith'])

    data['x' + post_fix] = np.cos(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['y' + post_fix] = np.sin(data['azimuth' + post_fix])*np.sin(data['zenith'+ post_fix])
    data['z' + post_fix] = np.cos(data['zenith' + post_fix])
    return data

def calculate_angular_difference(data, is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    data = convert_to_unit_vectors(data, post_fix)
    dotprod = (data['x']*data['x' + post_fix].values + data['y']*data['y'+ post_fix].values + data['z']*data['z'+ post_fix].values)
    norm_data = np.sqrt(data['x'+ post_fix]**2 + data['y'+ post_fix]**2 + data['z'+ post_fix]**2).values
    norm_truth = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2).values

    cos_angle = dotprod/(norm_data*norm_truth)

    return np.arccos(cos_angle).values*(360/(2*np.pi))

def calculate_width(data, target, is_retro = True):
    tracks = data.loc[data['track'] == 1, :]
    cascades = data.loc[data['track'] == 0, :]
    track_residual = calculate_residual(tracks, target, is_retro)
    cascade_residual = calculate_residual(cascades, target, is_retro)
    if target != 'angular_res':
        return (np.percentile(track_residual,84) - np.percentile(track_residual,16))/2, (np.percentile(cascade_residual,84) - np.percentile(cascade_residual,16))/2
    else:
        return np.percentile(track_residual,50), np.percentile(cascade_residual, 50)

def calculate_residual(data, target, is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    if target == 'energy':
        residual = ((np.log10(data[target + post_fix]) - np.log10(data[target]))/np.log10(data[target]))*100
    if target in ['azimuth', 'zenith']:
        residual = (data[target + post_fix] - data[target])*(360/(2*np.pi))
        residual = (residual + 180) % 360 -180
    if target == 'angular_res':
        residual = calculate_angular_difference(data, is_retro)
    if target == 'XYZ':
        residual = calculate_xyz_difference(data, is_retro)
    return residual

def calculate_bias(data, target, is_retro = True):
    tracks = data.loc[data['track'] == 1, :]
    cascades = data.loc[data['track'] == 0, :]
    return np.percentile(calculate_residual(tracks, target, is_retro),50), np.percentile(calculate_residual(cascades, target, is_retro),50)

def add_bias(statistics,data, target, is_retro = True):
    if is_retro:
        model = 'retro'
    else:
        model = 'dynedge'
    tracks, cascades = calculate_bias(data, target, is_retro)

    statistics[model + '_bias_'  + target +'_' + 'track'].append(tracks)
    statistics[model + '_bias_'  + target  +'_' +  'cascade'].append(cascades)
    return statistics

def add_width(statistics,data, target, is_retro = True):
    if is_retro:
        model = 'retro'
    else:
        model = 'dynedge'
    tracks, cascades = calculate_width(data, target, is_retro)
    try:
        statistics[model + '_width_' + target +'_' + 'track'].append(tracks)
        statistics[model + '_width_' + target +'_'  + 'cascade'].append(cascades)
    except:
        statistics[model + '_width_' + target +'_' + 'track'] = []
        statistics[model + '_width_' + target +'_'  + 'cascade'] = []
        statistics[model + '_width_' + target +'_' + 'track'].append(tracks)
        statistics[model + '_width_' + target +'_'  + 'cascade'].append(cascades)

    return statistics

def add_track_label(data):
    data['track'] = 0
    data['track'][(abs(data['pid']) == 14) & (data['interaction_type'] == 1)] = 1
    return data

def generate_file_name(plot_type):
    now = datetime.now()
    return plot_type  +'_' + '-'.join(str(now).split(' '))[0:16]+ '.pdf'

def move_old_plots(bias_filename, resolution_filename):
    files = os.listdir('plots')
    os.makedirs('plots/old', exist_ok = True)
    for file in files:
        if '.png' in file:
            if file not in [bias_filename, resolution_filename]:
                print('MOVING %s to old/%s'%(file,file))
                os.rename('plots/' + file, "plots/old/" + file)
    return

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data

def change_distribution_to_match_systematics(data):
    percent_v_e = 0.236917
    percent_v_u = 0.525125
    percent_v_t = 0.236917
    
    v_e = data.loc[abs(data['pid'] == 12),:]
    n_v_e = len(v_e)
    total_new_samples = n_v_e/percent_v_e
    n_v_u = int(total_new_samples*percent_v_u)
    n_v_t = int(total_new_samples*percent_v_t)

    v_u = data.loc[abs(data['pid'] == 14),:].sample(n_v_u)
    v_t = data.loc[abs(data['pid'] == 16),:].sample(n_v_t)

    data = v_e.append(v_u, ignore_index=True ).append(v_t, ignore_index=True )
    return data.sort_values('event_no').reset_index(drop = True)

def count_events_in_systematic_sets(data_folder):
    folders = os.listdir(data_folder)
    count = 0
    for folder in folders:
        if '0000' != folder:
            count += len(pd.read_csv(data_folder + '/' + folder + '/everything.csv' ))
    print('found %s events in %s systematic sets'%(count, len(folders)-1))
    return

def read_csv_and_make_statistics(data_folder):
    folders = os.listdir(data_folder)
    statistics = {'systematic': [],
                  'retro_bias_zenith_track': [],
                  'retro_bias_zenith_cascade': [],
                  'retro_bias_energy_track': [],
                  'retro_bias_energy_cascade': [],
                  'retro_bias_azimuth_track': [],
                  'retro_bias_azimuth_cascade': [],
                  'retro_width_zenith_cascade': [],
                  'retro_width_zenith_track': [],
                  'retro_width_energy_track': [],
                  'retro_width_energy_cascade': [],
                  'retro_width_angular_res_track': [],
                  'retro_width_angular_res_cascade': [],
                  'dynedge_bias_zenith_track': [],
                  'dynedge_bias_zenith_cascade': [],
                  'dynedge_bias_energy_track': [],
                  'dynedge_bias_energy_cascade': [],
                  'dynedge_bias_azimuth_track': [],
                  'dynedge_bias_azimuth_cascade': [],
                  'dynedge_width_zenith_track': [],
                  'dynedge_width_zenith_cascade': [],
                  'dynedge_width_energy_track': [],
                  'dynedge_width_energy_cascade': [],
                  'dynedge_width_angular_res_track': [],
                  'dynedge_width_angular_res_cascade': [],
                  'dynedge_signal_auc': [],
                  'dynedge_track_auc': [],
                  'retro_signal_auc': [],
                  'retro_track_auc': [],
                  'retro_bias_XYZ_track': [],
                  'retro_bias_XYZ_cascade': [],
                  'retro_width_XYZ_track': [],
                  'retro_width_XYZ_cascade': [],
                  'dynedge_bias_XYZ_track': [],
                  'dynedge_bias_XYZ_cascade': [],
                  'dynedge_width_XYZ_track': [],
                  'dynedge_width_XYZ_cascade': []}
    count = len(folders)-1
    for folder in folders:
        print('Reading %s. %s folders left.'%(folder, count))
        if '0000' != folder:
            data = pd.read_csv(data_folder + '/' + folder + '/' + 'everything.csv')
            statistics['dynedge_signal_auc'].append(calculate_auc(data, is_retro = False, target = 'neutrino'))
            statistics['retro_signal_auc'].append(calculate_auc(data, is_retro = True, target = 'neutrino'))
            data = remove_muons(data)
            statistics['systematic'].append(folder)
            statistics = add_bias(statistics, data, 'zenith', is_retro = True)
            statistics = add_bias(statistics, data, 'energy', is_retro = True)
            statistics = add_bias(statistics, data, 'azimuth', is_retro = True)
            statistics = add_bias(statistics, data, 'XYZ', is_retro = True)
            statistics = add_width(statistics,data, 'zenith', is_retro = True)
            statistics = add_width(statistics,data, 'energy', is_retro = True)
            statistics = add_width(statistics,data, 'angular_res', is_retro = True)
            statistics = add_width(statistics,data, 'XYZ', is_retro = True)

            statistics = add_bias(statistics, data, 'zenith', is_retro = False)
            statistics = add_bias(statistics, data, 'energy', is_retro = False)
            statistics = add_bias(statistics, data, 'azimuth', is_retro = False)
            statistics = add_bias(statistics, data, 'XYZ', is_retro = False)
            statistics = add_width(statistics,data, 'zenith', is_retro = False)
            statistics = add_width(statistics,data, 'energy', is_retro = False)
            statistics = add_width(statistics,data, 'angular_res', is_retro = False)
            statistics = add_width(statistics,data, 'XYZ', is_retro = False)

            statistics['dynedge_track_auc'].append(calculate_auc(data, is_retro = False, target = 'track'))
            statistics['retro_track_auc'].append(calculate_auc(data, is_retro = True, target = 'track'))
        else:
            data = pd.read_csv(data_folder + '/' + folder + '/' + 'reconstruction.csv')
            data = add_track_label(data)
            track_cascade = pd.read_csv(data_folder + '/' + folder + '/' + 'track_cascade.csv')
            signal = pd.read_csv(data_folder + '/' + folder + '/' + 'signal.csv')
            statistics['dynedge_signal_auc'].append(calculate_auc(signal, is_retro = False, target = 'neutrino'))
            statistics['retro_signal_auc'].append(calculate_auc(signal, is_retro = True, target = 'neutrino'))
            data = remove_muons(data)
            track_cascade = remove_muons(track_cascade)
            statistics['systematic'].append(folder)
            statistics = add_bias(statistics, data, 'zenith', is_retro = True)
            statistics = add_bias(statistics, data, 'energy', is_retro = True)
            statistics = add_bias(statistics, data, 'azimuth', is_retro = True)
            statistics = add_bias(statistics, data, 'XYZ', is_retro = True)
            statistics = add_width(statistics,data, 'zenith', is_retro = True)
            statistics = add_width(statistics,data, 'energy', is_retro = True)
            statistics = add_width(statistics,data, 'angular_res', is_retro = True)
            statistics = add_width(statistics,data, 'XYZ', is_retro = True)

            statistics = add_bias(statistics, data, 'zenith', is_retro = False)
            statistics = add_bias(statistics, data, 'energy', is_retro = False)
            statistics = add_bias(statistics, data, 'azimuth', is_retro = False)
            statistics = add_bias(statistics, data, 'XYZ', is_retro = False)
            statistics = add_width(statistics,data, 'zenith', is_retro = False)
            statistics = add_width(statistics,data, 'energy', is_retro = False)
            statistics = add_width(statistics,data, 'angular_res', is_retro = False)
            statistics = add_width(statistics,data, 'XYZ', is_retro = False)


            statistics['dynedge_track_auc'].append(calculate_auc(track_cascade, is_retro = False, target = 'track'))
            statistics['retro_track_auc'].append(calculate_auc(track_cascade, is_retro = True, target = 'track'))
        count = count -1 
    df = pd.DataFrame(statistics)
    df.to_csv('robustness_statistics.csv')
    return df

def calculate_rms(data, mode):
    if mode == 'RRI':
        x = (1 - data/data[0])*100
    if mode == 'bias':
        x = data - data[0]
    return np.sqrt(np.mean(x**2))

def print_rms_values(df):
    models = ['dynedge', 'retro']
    targets = ['zenith', 'energy', 'angular_res', 'track', 'signal', 'XYZ', 'azimuth']
    for target in targets:
        print('-----%s-----'%target)
        for model in models:
            if target not in ['signal', 'track']:
                if target != 'azimuth':
                    print('%s_width RMS track: %s'%(model,round(calculate_rms(df[model + '_width_' + target + '_track'], mode = 'RRI'),3)))
                    print('%s_width RMS cascade: %s'%(model,round(calculate_rms(df[model + '_width_' + target  + '_cascade'], mode = 'RRI'),3)))
                if target != 'angular_res':
                    print('%s_bias RMS track: %s'%(model,round(calculate_rms(df[model + '_bias_' + target + '_track'], mode = 'bias'),3)))
                    print('%s_bias RMS cascade: %s'%(model,round(calculate_rms(df[model + '_bias_' + target + '_cascade'], mode = 'bias'),3)))
            else:
                print('%s_AUC RMS: %s'%(model,round(calculate_rms(df[model + '_' + target + '_auc'], mode = 'RRI'),3)))

    return

def make_robustness_plots(data_folder, from_csv = False):
    if from_csv:
        df = read_csv_and_make_statistics(data_folder).sort_values('systematic').reset_index(drop = True)
    else:
        df = pd.read_csv('robustness_statistics.csv', dtype={'systematic': object}).sort_values('systematic').reset_index(drop = True)
    print_rms_values(df)

    width = 2*3.176
    height = 2*3.176
    

    ### RESOLUTION
    fig, ax = plt.subplots(3,1, figsize = (8,8))
    
    ax[0].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_zenith_track']/df['dynedge_width_zenith_track'][0])*100,marker='o', markeredgecolor='black', label = 'GNN Track', color = 'blue')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_energy_track']/df['dynedge_width_energy_track'][0])*100,marker='o', markeredgecolor='black', label = 'GNN Track', color = 'blue')
    ax[0].errorbar(np.arange(0,len(df)), (1- df['retro_width_zenith_track']/df['retro_width_zenith_track'][0])*100,marker='o', markeredgecolor='black', label = 'RetroReco Track', color = 'orange')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['retro_width_energy_track']/df['retro_width_energy_track'][0])*100,marker='o', markeredgecolor='black', label = 'RetroReco Track', color = 'orange')
    
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['dynedge_width_angular_res_track']/df['dynedge_width_angular_res_track'][0])*100,marker='o', markeredgecolor='red', label = 'GNN Track', color = 'blue')
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['retro_width_angular_res_track']/df['retro_width_angular_res_track'][0])*100,marker='o', markeredgecolor='red', label = 'RetroReco Track', color = 'orange')
    
    ax[0].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_zenith_cascade']/df['dynedge_width_zenith_cascade'][0])*100,marker='o', markeredgecolor='red', ls = '--', label = 'GNN Cascade', color = 'blue')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_energy_cascade']/df['dynedge_width_energy_cascade'][0])*100,marker='o', markeredgecolor='red', ls = '--', label = 'GNN Cascade', color = 'blue')
    ax[0].errorbar(np.arange(0,len(df)), (1- df['retro_width_zenith_cascade']/df['retro_width_zenith_cascade'][0])*100,marker='o', markeredgecolor='red', ls = '--', label = 'RetroReco Cascade', color = 'orange')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['retro_width_energy_cascade']/df['retro_width_energy_cascade'][0])*100,marker='o', markeredgecolor='red', ls = '--', label = 'RetroReco Cascade', color = 'orange')
    
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['dynedge_width_angular_res_cascade']/df['dynedge_width_angular_res_cascade'][0])*100,marker='o', ls = '--' ,markeredgecolor='red', label = 'GNN Cascade', color = 'blue')
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['retro_width_angular_res_cascade']/df['retro_width_angular_res_cascade'][0])*100,marker='o', ls = '--' ,markeredgecolor='red', label = 'RetroReco Cascade', color = 'orange')
    


    ax[1].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    ax[0].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    ax[2].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)

    #ax[1].set_ylim([-12,12])
    #ax[0].set_ylim([-12,12])
    #ax[2].set_ylim([-12,12])
    ax[1].grid()
    ax[0].grid()
    ax[2].grid()
    #ax[1].legend()
    ax[0].legend()
    ax[0].tick_params(bottom=True,labelbottom=False)
    ax[1].tick_params(bottom=True,labelbottom=False)
    
    #ax[0].set_ylabel('Variation w.r.t. Nominal [%]', size = 11)
    #ax[1].set_ylabel('Variation w.r.t. Nominal [%]', size = 11)
    #ax[2].set_ylabel('Variation w.r.t. Nominal [%]', size = 11)
    fig.text(0.04, 0.5, 'Resolution Improvement [%]', va='center', rotation='vertical', size = 12)
    ax[1].set_xticks(np.arange(0,len(df)))
    ax[0].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticklabels(df['systematic'].values.tolist(), rotation = 25, fontsize = 8)
    ax[2].set_xlabel('Systematic Set', size = 13)
    #fig.suptitle('Robustness of Resolution', size = 8)    
    ax[0].set_title('Zenith')
    ax[1].set_title('Energy')
    ax[2].set_title('Direction')
    resolution_filename = generate_file_name(plot_type = 'resolution')
    print(resolution_filename)
    fig.savefig('plots/' + resolution_filename)

    
    #### BIAS
    fig, ax = plt.subplots(3,1, figsize = (8,8))
    
    ax[0].errorbar(np.arange(0,len(df)), df['dynedge_bias_zenith_track'] - df['dynedge_bias_zenith_track'][0],marker='o', markeredgecolor='black', label = 'GNN Track', color = 'blue')
    ax[1].errorbar(np.arange(0,len(df)), df['dynedge_bias_energy_track'] - df['dynedge_bias_energy_track'][0],marker='o', markeredgecolor='black', label = 'GNN Track', color = 'blue')
    ax[0].errorbar(np.arange(0,len(df)), df['retro_bias_zenith_track']   - df['retro_bias_zenith_track'][0],marker='o', markeredgecolor='black', label = 'RetroReco Track', color = 'orange')
    ax[1].errorbar(np.arange(0,len(df)), df['retro_bias_energy_track']   - df['retro_bias_energy_track'][0],marker='o', markeredgecolor='black', label = 'RetroReco Track', color = 'orange')
    
    ax[2].errorbar(np.arange(0,len(df)), df['dynedge_bias_azimuth_track']-df['dynedge_bias_azimuth_track'][0],marker='o', markeredgecolor='black', label = 'GNN Track', color = 'blue')
    ax[2].errorbar(np.arange(0,len(df)), df['retro_bias_azimuth_track']  -df['retro_bias_azimuth_track'][0],marker='o', markeredgecolor='black', label = 'RetroReco Track', color = 'orange')
    
    ax[0].errorbar(np.arange(0,len(df)), df['dynedge_bias_zenith_cascade'] - df['dynedge_bias_zenith_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'GNN Cascade', color = 'blue')
    ax[1].errorbar(np.arange(0,len(df)), df['dynedge_bias_energy_cascade'] - df['dynedge_bias_energy_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'GNN Cascade', color = 'blue')
    ax[0].errorbar(np.arange(0,len(df)), df['retro_bias_zenith_cascade']   - df['retro_bias_zenith_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'RetroReco Cascade', color = 'orange')
    ax[1].errorbar(np.arange(0,len(df)), df['retro_bias_energy_cascade']   - df['retro_bias_energy_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'RetroReco Cascade', color = 'orange')
    
    ax[2].errorbar(np.arange(0,len(df)), df['dynedge_bias_azimuth_cascade']-df['dynedge_bias_azimuth_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'GNN Cascade', color = 'blue')
    ax[2].errorbar(np.arange(0,len(df)), df['retro_bias_azimuth_cascade']  -df['retro_bias_azimuth_cascade'][0],marker='o', ls = '--' ,markeredgecolor='red', label = 'RetroReco Cascade', color = 'orange')
    

    ax[1].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    ax[0].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    ax[2].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    #ax[1].set_ylim([-12,12])
    #ax[0].set_ylim([-12,12])
    ax[1].grid()
    ax[0].grid()
    ax[2].grid()
    #ax[1].legend()
    ax[0].legend()
    ax[0].tick_params(bottom=True,labelbottom=False)
    ax[1].tick_params(bottom=True,labelbottom=False)
    
    ax[0].set_ylabel('[Deg.]', size = 12)
    ax[2].set_ylabel('[Deg.]', size = 12)
    ax[1].set_ylabel('[%]', size = 12)
    ax[1].set_xticks(np.arange(0,len(df)))
    ax[0].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticklabels(df['systematic'].values.tolist(), rotation = 25)
    ax[2].set_xlabel('Systematic Set', size = 13)
    fig.suptitle('Bias Variation', size = 20)
    ax[0].set_title('Zenith')
    ax[1].set_title('Energy')
    ax[2].set_title('Azimuth')
    
    #fig.text(0.04, 0.5, 'Bias Variation', va='center', rotation='vertical', size = 15)

    bias_filename = generate_file_name(plot_type = 'bias')
    fig.savefig('plots/' + bias_filename)
    move_old_plots(bias_filename, resolution_filename)

    #### CLASSIFICATION
    fig, ax = plt.subplots(2,1, figsize = (8,8))
    
    ax[0].errorbar(np.arange(0,len(df)), (-1 + df['dynedge_signal_auc']/df['dynedge_signal_auc'][0])*100,marker='o', markeredgecolor='black', label = 'GCN')
    ax[1].errorbar(np.arange(0,len(df)), (-1 + df['dynedge_track_auc']/df['dynedge_track_auc'][0])*100,marker='o', markeredgecolor='black', label = 'GCN')
    ax[0].errorbar(np.arange(0,len(df)), (-1 + df['retro_signal_auc']/df['retro_signal_auc'][0])*100,marker='o', markeredgecolor='black', label = 'BDT')
    ax[1].errorbar(np.arange(0,len(df)), (-1 + df['retro_track_auc']/df['retro_track_auc'][0])*100,marker='o', markeredgecolor='black', label = 'BDT')
    
 

    ax[1].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    ax[0].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)

    ax[1].set_ylim([-5,5])
    ax[0].set_ylim([-5,5])
    ax[1].grid()
    ax[0].grid()
    #ax[1].legend()
    ax[0].legend()
    ax[0].tick_params(bottom=True,labelbottom=False)
    
    ax[1].set_xticks(np.arange(0,len(df)))
    ax[0].set_xticks(np.arange(0,len(df)))
    ax[1].set_xticklabels(df['systematic'].values.tolist(), rotation = 25)
    ax[1].set_xlabel('Systematic Set', size = 13)
    fig.suptitle('Robustness of AUC', size = 20)
    ax[0].set_title('Signal')
    ax[1].set_title('Track/Cascade')
 
    
    fig.text(0.04, 0.5, 'AUC Variation [%]', va='center', rotation='vertical', size = 15)
    bias_filename = generate_file_name(plot_type = 'classification')
    fig.savefig('plots/' + bias_filename)
    return 

if __name__ == '__main__':
    #data_nominal = 'plot_data_nominal_model.csv'
    data_folder = '/remote/ceph/user/o/oersoe/paper_data/data'
    #count_events_in_systematic_sets(data_folder)
    make_robustness_plots(data_folder, from_csv = False)
