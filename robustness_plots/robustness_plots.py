import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import stats
import os
import matplotlib as mpl
import multiprocessing
import sqlite3
mpl.use('pdf')
plt.rc('font', family='serif')
#mpl.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})
#mpl.rcParams['text.usetex'] = True
from pandas.core.algorithms import diff
import pickle

def add_auc(statistics, data, is_retro, target):
    if is_retro:
        model = 'retro_'
    else:
        model = 'dynedge_'
    if isinstance(data, dict):
        for key in data.keys():
            if target == 'neutrino':
                if 'signal' in key.lower():
                    if '0' in key.lower():
                        name = model + '%s_%s_auc'%(target,'nominal')
                    else:
                        name = model + '%s_%s_auc'%(target,'sys')
                    try:
                        statistics[name].append(calculate_auc(data[key], is_retro = is_retro, target = target))
                    except:
                        statistics[name] = []
                        statistics[name].append(calculate_auc(data[key], is_retro = is_retro, target = target))
            if target == 'track':
                if 'track' in key.lower():
                    if '0' in key.lower():
                        name = model + '%s_%s_auc'%(target,'nominal')
                    else:
                        name = model + '%s_%s_auc'%(target,'sys')
                    try:
                        statistics[name].append(calculate_auc(data[key], is_retro = is_retro, target = target))
                    except:
                        statistics[name] = []
                        statistics[name].append(calculate_auc(data[key], is_retro = is_retro, target = target))
        return statistics

    else:
        try:
            statistics[model + '%s_auc'%target].append(calculate_auc(data, is_retro = is_retro, target = 'neutrino'))
        except:
            statistics[model + '%s_auc'%target] = []
            statistics[model + '%s_auc'%target].append(calculate_auc(data, is_retro = is_retro, target = 'neutrino'))
        return statistics


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
    if target == 'azimuth':
        laplace = True
    else:
        laplace = False
    tracks = data.loc[(abs(data['pid']) == 14) & (data['interaction_type'] == 1), :].reset_index(drop = True)
    cascades = data.loc[(abs(data['pid']) != 14) | (data['interaction_type'] != 1), :].reset_index(drop = True)
    track_residual = calculate_residual(tracks, target, is_retro)
    cascade_residual = calculate_residual(cascades, target, is_retro)
    if target not in ['XYZ', 'angular_res']:
        track_error = add_width_error(track_residual, laplace = laplace)
        cascade_error = add_width_error(cascade_residual, laplace = laplace)
        track_width = (np.percentile(track_residual,84) - np.percentile(track_residual,16))/2
        cascade_width = (np.percentile(cascade_residual,84) - np.percentile(cascade_residual,16))/2
        return track_width, cascade_width, track_error, cascade_error
    else:
        track_error = add_50th_error(track_residual, laplace = laplace)
        cascade_error = add_50th_error(cascade_residual, laplace = laplace)
        track_width = np.percentile(track_residual,50)
        cascade_width =  np.percentile(cascade_residual, 50)
        return track_width, cascade_width, track_error, cascade_error

def calculate_residual(data, target, is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    if target == 'energy':
        #residual = ((np.log10(data[target + post_fix]) - np.log10(data[target]))/np.log10(data[target]))*100
        residual = ((data[target + post_fix] - data[target])/(data[target]))*100
    if target in ['azimuth', 'zenith']:
        residual = (data[target + post_fix] - data[target])*(360/(2*np.pi))
        residual = (residual + 180) % 360 -180
    if target == 'angular_res':
        residual = calculate_angular_difference(data, is_retro)
    if target == 'XYZ':
        residual = calculate_xyz_difference(data, is_retro)
    return residual

def calculate_bias(data, target, is_retro = True):
    if target == 'azimuth':
        laplace = True
    else:
        laplace = False
    tracks = data.loc[data['track'] == 1, :].reset_index(drop = True)
    cascades = data.loc[data['track'] == 0, :].reset_index(drop = True)
    track_residual = calculate_residual(tracks, target, is_retro)
    cascade_residual = calculate_residual(cascades, target, is_retro)
    track_error = add_50th_error(track_residual, laplace = laplace)
    cascade_error = add_50th_error(cascade_residual, laplace = laplace)
    return np.percentile(track_residual,50), np.percentile(cascade_residual,50), track_error, cascade_error

def parallel_50th_error(settings):
    queue, n_samples, batch_size, diff = settings
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        new_sample = rng.choice(diff, size = batch_size, replace = True)
        queue.put(np.percentile(new_sample,50))
    multiprocessing.current_process().close()

def parallel_width_error(settings):
    queue, n_samples, batch_size, diff = settings
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        new_sample = rng.choice(diff, size = batch_size, replace = True)
        queue.put([np.percentile(new_sample,84),np.percentile(new_sample,16)])
    multiprocessing.current_process().close()


def add_50th_error(diff, laplace = False):
    if __name__ == '__main__':
        manager = multiprocessing.Manager()
        q = manager.Queue()
        total_samples = 10000
        batch_size = len(diff)
        n_workers = 100
        samples_pr_worker = int(total_samples/n_workers)
        settings = []
        for i in range(n_workers):
            settings.append([q, samples_pr_worker, batch_size, diff])
        p = multiprocessing.Pool(processes = len(settings))
        async_result = p.map_async(parallel_50th_error, settings)
        p.close()
        p.join()
        p50 = []
        queue_empty = q.empty()
        while(queue_empty == False):
            queue_empty = q.empty()
            if queue_empty == False:
                p50.append(q.get())
        return np.std(p50)

def add_width_error(diff, laplace = False):
    if __name__ == '__main__':
        manager = multiprocessing.Manager()
        q = manager.Queue()
        total_samples = 10000
        batch_size = len(diff)
        n_workers = 100
        samples_pr_worker = int(total_samples/n_workers)
        settings = []
        for i in range(n_workers):
            settings.append([q, samples_pr_worker, batch_size, diff])
        p = multiprocessing.Pool(processes = len(settings))
        async_result = p.map_async(parallel_width_error, settings)
        p.close()
        p.join()
        p16 = []
        p84 = []
        queue_empty = q.empty()
        while(queue_empty == False):
            queue_empty = q.empty()
            if queue_empty == False:
                item = q.get()
                p84.append(item[0])
                p16.append(item[1])
    return np.sqrt(np.std(p16)**2 + np.std(p84)**2)

def add_bias(statistics,data, target, is_retro = True):
    if is_retro:
        model = 'retro'
    else:
        model = 'dynedge'
    if isinstance(data, dict):
        for key in data.keys():
            if 'reco' in key.lower():
                if '0' in key.lower():
                    name = model + '_%s_%s_bias'%(target,'nominal')
                else:
                    name = model + '_%s_%s_bias'%(target,'sys')
                tracks, cascades, track_error, cascade_error = calculate_bias(data[key], target, is_retro)
                try:
                    statistics[name + '_' + 'track'].append(tracks)
                    statistics[name +'_' +  'cascade'].append(cascades)
                    statistics[name +'_' + 'track_error'].append(track_error)
                    statistics[name +'_' +  'cascade_error'].append(cascade_error)
                except:
                    statistics[name +'_' + 'track'] = []
                    statistics[name  +'_' +  'cascade']= []
                    statistics[name +'_' + 'track_error']= []
                    statistics[name +'_' +  'cascade_error']= []

                    statistics[name  +'_' + 'track'].append(tracks)
                    statistics[name  +'_' +  'cascade'].append(cascades)
                    statistics[name  +'_' + 'track_error'].append(track_error)
                    statistics[name  +'_' +  'cascade_error'].append(cascade_error)
        return statistics
    else:
        tracks, cascades, track_error, cascade_error = calculate_bias(data, target, is_retro)
        try:
            statistics[model + '_bias_'  + target +'_' + 'track'].append(tracks)
            statistics[model + '_bias_'  + target  +'_' +  'cascade'].append(cascades)
            statistics[model + '_bias_'  + target +'_' + 'track_error'].append(track_error)
            statistics[model + '_bias_'  + target  +'_' +  'cascade_error'].append(cascade_error)
        except:
            statistics[model + '_bias_'  + target +'_' + 'track'] = []
            statistics[model + '_bias_'  + target  +'_' +  'cascade']= []
            statistics[model + '_bias_'  + target +'_' + 'track_error']= []
            statistics[model + '_bias_'  + target  +'_' +  'cascade_error']= []

            statistics[model + '_bias_'  + target +'_' + 'track'].append(tracks)
            statistics[model + '_bias_'  + target  +'_' +  'cascade'].append(cascades)
            statistics[model + '_bias_'  + target +'_' + 'track_error'].append(track_error)
            statistics[model + '_bias_'  + target  +'_' +  'cascade_error'].append(cascade_error)
        return statistics




def add_width(statistics,data, target, is_retro = True):
    if is_retro:
        model = 'retro'
    else:
        model = 'dynedge'
    if isinstance(data,dict):
        for key in data.keys():
            if 'reco' in key.lower():
                if '0' in key.lower():
                    name = model + '_%s_%s_width'%(target,'nominal')
                else:
                    name = model + '_%s_%s_width'%(target,'sys')
                tracks, cascades, track_error, cascade_error = calculate_width(data[key], target, is_retro)
                try:
                    statistics[name +'_' + 'track'].append(tracks)
                    statistics[name +'_'  + 'cascade'].append(cascades)
                    statistics[name +'_' + 'track_error'].append(track_error)
                    statistics[name +'_'  + 'cascade_error'].append(cascade_error)
                except:
                    statistics[name +'_' + 'track'] = []
                    statistics[name +'_'  + 'cascade'] = []
                    statistics[name +'_' + 'track'].append(tracks)
                    statistics[name +'_'  + 'cascade'].append(cascades)
                    statistics[name +'_' + 'track_error'] = []
                    statistics[name +'_'  + 'cascade_error'] = []
                    statistics[name +'_' + 'track_error'].append(track_error)
                    statistics[name +'_'  + 'cascade_error'].append(cascade_error)
        return statistics
            
    else:
        tracks, cascades, track_error, cascade_error = calculate_width(data, target, is_retro)
        try:
            statistics[model + '_width_' + target +'_' + 'track'].append(tracks)
            statistics[model + '_width_' + target +'_'  + 'cascade'].append(cascades)
            statistics[model + '_width_' + target +'_' + 'track_error'].append(track_error)
            statistics[model + '_width_' + target +'_'  + 'cascade_error'].append(cascade_error)
        except:
            statistics[model + '_width_' + target +'_' + 'track'] = []
            statistics[model + '_width_' + target +'_'  + 'cascade'] = []
            statistics[model + '_width_' + target +'_' + 'track'].append(tracks)
            statistics[model + '_width_' + target +'_'  + 'cascade'].append(cascades)
            statistics[model + '_width_' + target +'_' + 'track_error'] = []
            statistics[model + '_width_' + target +'_'  + 'cascade_error'] = []
            statistics[model + '_width_' + target +'_' + 'track_error'].append(track_error)
            statistics[model + '_width_' + target +'_'  + 'cascade_error'].append(cascade_error)
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
        if '.pdf' in file:
            if file not in [bias_filename, resolution_filename, 'bar_' + resolution_filename, 'bar_' + bias_filename ]:
                print('MOVING %s to old/%s'%(file,file))
                os.rename('plots/' + file, "plots/old/" + file)
    return

def remove_muons(data):
    if isinstance(data, dict):
        for entry in data.keys():
            data[entry] = data[entry].loc[abs(data[entry]['pid'] != 13), :]
    else:
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

def get_overlap(df,folder, out = None, tag = None):
    keys = ['%s_overlap'%folder, 'track_overlap', 'signal_overlap', 'reco_overlap']
    if out == None:
        out = {}
    c = 0
    for key in keys:
        if key in df.columns:
            if tag == 'sys':
                out[key] = df.loc[df[key] == 1,:].reset_index(drop = True)
                out[key] =  add_track_label(out[key])
            else:
                print(tag + '_' + key)
                out[tag + '_' + key] = df.loc[df[key] == 1,:].reset_index(drop = True)
                out[tag + '_' + key] =  add_track_label(out[tag + '_' + key])    
            c +=1
    if c == 0:
        print('WARNING: Nothing for %s'%folder)
    return out
    
    

def get_nominal_overlap(data_folder,folder):
    path = '/mnt/scratch/rasmus_orsoe/paper_data/data_with_overlap_labels/0000/'

    data  = get_overlap(pd.read_csv(path + 'signal.csv'), folder, tag = 'signal') 
    data = get_overlap(pd.read_csv(path + 'track_cascade.csv'), folder, data, tag = 'track')
    data = get_overlap(pd.read_csv(path + 'reconstruction.csv'), folder, data, tag = 'reco') #get_overlap(nominal_reco_csv, nominal)
    data = get_overlap(pd.read_csv(data_folder + '/' + folder + '/' + 'everything.csv', ), folder, data, tag = 'sys') #get_overlap(sys_csv, sys)

    return data

def get_overlapping_events(data):
    overlapping_events = pd.read_csv('/home/iwsatlas1/oersoe/phd/paper/robustness/matching_events/overlapping_events_sql.csv')
    index = []
    for i in range(0,len(overlapping_events)):
        match = data.loc[(data['zenith'] == overlapping_events['zenith'][i]) & (data['energy'] == overlapping_events['energy'][i]) & (data['azimuth'] == overlapping_events['azimuth'][i]),:]
        index.append(match.index[0])
    return data.loc[index,:].reset_index(drop = True)

def print_keys(statistics):
    c = []
    for key in statistics.keys():
        if 'angular' in key:
            c.append(key)
    print(c)
    return

def parallel_statistics(settings):
    folder, data_folder, statistics, queue = settings 
    if '0000' != folder:  
        data = get_nominal_overlap(data_folder,folder)
        statistics = add_auc(statistics,data, is_retro = False, target = 'neutrino')
        statistics = add_auc(statistics,data, is_retro = True, target = 'neutrino')
        data = remove_muons(data)
        statistics['systematic'].append(folder)
        statistics = add_width(statistics,data, 'angular_res', is_retro = True)
        statistics = add_width(statistics,data, 'angular_res', is_retro = False)
        statistics = add_bias(statistics, data, 'zenith', is_retro = True)
        statistics = add_bias(statistics, data, 'energy', is_retro = True)
        statistics = add_bias(statistics, data, 'azimuth', is_retro = True)
        statistics = add_bias(statistics, data, 'XYZ', is_retro = True)
        statistics = add_width(statistics,data, 'zenith', is_retro = True)
        statistics = add_width(statistics,data, 'energy', is_retro = True)
        statistics = add_width(statistics,data, 'azimuth', is_retro = True)
        statistics = add_width(statistics,data, 'XYZ', is_retro = True)
        statistics = add_bias(statistics, data, 'zenith', is_retro = False)
        statistics = add_bias(statistics, data, 'energy', is_retro = False)
        statistics = add_bias(statistics, data, 'azimuth', is_retro = False)
        statistics = add_bias(statistics, data, 'XYZ', is_retro = False)
        statistics = add_width(statistics,data, 'zenith', is_retro = False)
        statistics = add_width(statistics,data, 'energy', is_retro = False)
        statistics = add_width(statistics,data, 'azimuth', is_retro = False)
        statistics = add_width(statistics,data, 'XYZ', is_retro = False)
        statistics = add_auc(statistics, data, is_retro = False, target = 'track')
        statistics = add_auc(statistics, data, is_retro = True, target = 'track')
        queue.put(statistics)
    print('%s Done'%folder)
    return

def read_csv_and_make_statistics(data_folder, overlap_only = False, relative_overlap = False):
    from copy import deepcopy
    folders = os.listdir(data_folder)
    statistics = {'systematic': [],
                  'dynedge_signal_auc': [],
                  'dynedge_track_auc': [],
                  'retro_signal_auc': [],
                  'retro_track_auc': []}
    queue = multiprocessing.Queue()
    jobs = []
    for folder in folders:
        #settings.append([folder, data_folder, deepcopy(statistics), queue])
        p = multiprocessing.Process(target = parallel_statistics, args = ([folder, data_folder, statistics, queue],))
        p.start()
        jobs.append(p)
    
    #for job in jobs:
    #    print('hi')
    #    job.join()
    print('jobs sent')
    is_first = True
    c = 0
    stop = False
    while stop == False:
        if queue.empty() == False:
            print('getting item')
            item = queue.get()
            print('got item')
            if is_first:
                print('%s / 20'%c)
                statistics = item
                is_first = False
                c+=1
            else:
                print('%s / 20'%c)
                for key in item.keys():
                    for element in item[key]:
                        try:
                            statistics[key].append(element)
                        except:
                            statistics[key] = []
                            statistics[key].append(element)
                c+=1
            print(len(statistics))
            if c == 20:
                stop = True

    import pickle
    with open('robustness_statistics_rel_overlapv2.pkl', 'wb') as file:
        pickle.dump(statistics, file)
    return statistics

def calculate_rms(df, target, model, key, signature):
    if key == 'width':
        x = (1- df['%s_%s_sys_%s_%s'%(model, target,key,signature)]/df['%s_%s_nominal_%s_%s'%(model, target,key,signature)])*100
    if key == 'bias':
        x = df['%s_%s_sys_%s_%s'%(model, target,key,signature)] - df['%s_%s_nominal_%s_%s'%(model, target,key,signature)]

    if key == 'auc':
        x = (1- df['%s_%s_sys_%s'%(model, target,key)]/df['%s_%s_nominal_%s'%(model, target,key)])*100
    return np.sqrt(np.mean(x**2))

def print_rms_values(df):
    models = ['dynedge', 'retro']
    targets = ['zenith', 'energy', 'azimuth', 'track', 'neutrino', 'XYZ', 'azimuth', 'angular_res']
    for target in targets:
        print('-----%s-----'%target)
        for model in models:
            if target not in ['neutrino', 'track']:
                if target != 'azimuth':
                    print('%s_width RMS track: %s'%(model,round(calculate_rms(df, target, model, key = 'width', signature = 'track'),3)))
                    print('%s_width RMS cascade: %s'%(model,round(calculate_rms(df, target, model, key = 'width', signature = 'cascade'),3)))
                if target != 'angular_res':
                    print('%s_bias RMS track: %s'%(model,round(calculate_rms(df, target, model, key = 'bias', signature = 'track'),3)))
                    print('%s_bias RMS cascade: %s'%(model,round(calculate_rms(df, target, model, key = 'bias', signature = 'cascade'),3)))
            else:
                print('%s_AUC RMS: %s'%(model,round(calculate_rms(df, target, model, key = 'auc', signature = 'cascade'),3)))

    return

def replace_values(list_to_replace, item_to_replace, item_to_replace_with):
    return [item_to_replace_with if item == item_to_replace else item for item in list_to_replace]


def rename_sets(set_values):
    ## DOM EFF
    set_values = replace_values(set_values, '0000', '1')
    set_values = replace_values(set_values, '0001', '-10%')
    set_values = replace_values(set_values, '0002', '-5%')
    set_values = replace_values(set_values, '0003', '+5%')
    set_values = replace_values(set_values, '0004', '+10%')

    ## MIXED PROFILES
    set_values = replace_values(set_values, '0100', '(1,-0.07,-0.11)')
    set_values = replace_values(set_values, '0101', '(1,-0.48,-0.02)')
    set_values = replace_values(set_values, '0102', '(1,0.28,-0.08)')
    set_values = replace_values(set_values, '0103', '(1,0.11,0.004)')
    set_values = replace_values(set_values, '0104', '(0.88,-0.05,-0.05)')
    set_values = replace_values(set_values, '0105', '(0.93,-0.37,0.04)')
    set_values = replace_values(set_values, '0106', '(0.97,0.30,-0.04)')
    set_values = replace_values(set_values, '0107', '(1.03,0.12,-0.11)')
    set_values = replace_values(set_values, '0109', '(1.12,-0.31,-0.08)')
    set_values = replace_values(set_values, '0150', '(1,-1.3,0.15)')
    set_values = replace_values(set_values, '0151', '(1,-1,-0.1)')
    set_values = replace_values(set_values, '0152', '(1,0.5,0.15)')

    ## BULKICE
    set_values = replace_values(set_values, '0500', '(0.5, 0.5)')
    set_values = replace_values(set_values, '0501', '(0.5, -0.5)')
    set_values = replace_values(set_values, '0502', '(-0.5, 0.5)')
    set_values = replace_values(set_values, '0503', '(-0.5, -0.5)')
    return set_values

def relative_improvement_error(w1,w2, w1_sigma,w2_sigma):
    w1 = np.array(w1)
    w1_sigma = np.array(w1_sigma)
    #sigma = np.sqrt((np.array(w1_sigma)/np.array(w1))**2 + (np.array(w2_sigma)/np.array(w2))**2)
    sigma = np.sqrt(((1/w2)*w1_sigma)**2  + ((w1/w2**2)*w2_sigma)**2)
    return sigma

def bias_variation_error(w1,w2, w1_sigma,w2_sigma):
    sigma = np.sqrt((np.array(w1_sigma))**2 + (np.array(w2_sigma))**2)
    #sigma = np.sqrt(((1/w2)*w1_sigma)**2  + ((w1/w2**2)*w2_sigma)**2)
    return sigma

def make_robustness_plots(data_folder, binned = True, from_csv = False, overlap_only = False, relative_overlap = True):
    if from_csv:
        df = read_csv_and_make_statistics(data_folder, overlap_only, relative_overlap)
    else:
        file = '/home/iwsatlas1/oersoe/phd/paper/paper_data/plots/robustness_statistics_rel_overlapv2.pkl'
        with open(file, 'rb') as file:
            df = pickle.load(file)
    remove_these = []
    for key in df.keys():
        print(key)
        if len(df[key]) == 0:
            remove_these.append(key)
    for key in remove_these:
        df.pop(key)
    df = pd.DataFrame(df).sort_values('systematic').reset_index(drop = True)
    
    print_rms_values(df)

    width = 2*3.176
    height = 3*3.176
    
    set_values = df['systematic'].values.tolist()
    set_values = rename_sets(set_values)

    ### RESOLUTION BAR PLOT
    fig, ax = plt.subplots(3,1, figsize = (8,10), sharex=  True)
    bar_width = 0.15*3
    fmt = '.'
    capsize = 4
    alpha_track_gnn = 1
    alpha_cascade_gnn = 0.5
    alpha_track_retro = 1
    alpha_cascade_retro = 0.5

    gnn_error_color_track = 'tab:blue'
    gnn_error_color_cascade = 'tab:blue'

    retro_error_color_track = 'tab:orange'
    retro_error_color_cascade = 'tab:orange'
    errorbars = True
    x = np.arange(0,len(df)*3,3)
    ax[0].bar(x - (3/2)*bar_width, (1- df['dynedge_zenith_sys_width_track']/df['dynedge_zenith_nominal_width_track'])*100,bar_width,  label = 'GraphNet Track', color = 'tab:blue', alpha = 1)
    ax[1].bar(x - (3/2)*bar_width , (1- df['dynedge_energy_sys_width_track']/df['dynedge_energy_nominal_width_track'])*100,bar_width, color = 'tab:blue', alpha = 1)
    ax[0].bar(x - bar_width/2, (1- df['retro_zenith_sys_width_track']/df['retro_zenith_nominal_width_track'])*100,bar_width, label = 'Retro Track ', color = 'tab:orange', alpha = 1)
    ax[1].bar(x - bar_width/2, (1- df['retro_energy_sys_width_track']/df['retro_energy_nominal_width_track'])*100, bar_width, color = 'tab:orange', alpha = 1)
    
    ax[2].bar(x - (3/2)*bar_width, (1 - df['dynedge_angular_res_sys_width_track']/df['dynedge_angular_res_nominal_width_track'])*100, bar_width, color = 'tab:blue', alpha = 1)
    ax[2].bar(x - bar_width/2, (1 - df['retro_angular_res_sys_width_track']/df['retro_angular_res_nominal_width_track'])*100, bar_width, color = 'tab:orange', alpha = 1)
    
    ax[0].bar(x + bar_width/2, (1- df['dynedge_zenith_sys_width_cascade']/df['dynedge_zenith_nominal_width_cascade'])*100,bar_width, label = 'GraphNet Cascade', color = 'tab:blue', alpha = 0.5)
    ax[1].bar(x + bar_width/2, (1- df['dynedge_energy_sys_width_cascade']/df['dynedge_energy_nominal_width_cascade'])*100,bar_width, color = 'tab:blue', alpha = 0.5)
    ax[0].bar(x + (3/2)*bar_width, (1- df['retro_zenith_sys_width_cascade']/df['retro_zenith_nominal_width_cascade'])*100,bar_width, label = ' Retro Cascade ', color = 'tab:orange', alpha = 0.5)
    ax[1].bar(x + (3/2)*bar_width, (1- df['retro_energy_sys_width_cascade']/df['retro_energy_nominal_width_cascade'])*100,bar_width, color = 'tab:orange', alpha = 0.5)
    
    ax[2].bar(x + bar_width/2, (1 - df['dynedge_angular_res_sys_width_cascade']/df['dynedge_angular_res_nominal_width_cascade'])*100,bar_width, color = 'tab:blue', alpha = 0.5)
    ax[2].bar(x + (3/2)*bar_width, (1 - df['retro_angular_res_sys_width_cascade']/df['retro_angular_res_nominal_width_cascade'])*100, bar_width, color = 'tab:orange', alpha = 0.5)

    if errorbars == True:
        ax[0].errorbar(x - (3/2)*bar_width, (1- df['dynedge_zenith_sys_width_track']/df['dynedge_zenith_nominal_width_track'])*100, relative_improvement_error(df['dynedge_zenith_sys_width_track'],df['dynedge_zenith_nominal_width_track'],df['dynedge_zenith_sys_width_track_error'],df['dynedge_zenith_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color = gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[1].errorbar(x - (3/2)*bar_width , (1- df['dynedge_energy_sys_width_track']/df['dynedge_energy_nominal_width_track'])*100,relative_improvement_error(df['dynedge_energy_sys_width_track'],df['dynedge_energy_nominal_width_track'],df['dynedge_energy_sys_width_track_error'],df['dynedge_energy_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color = gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[0].errorbar(x - bar_width/2, (1- df['retro_zenith_sys_width_track']/df['retro_zenith_nominal_width_track'])*100,relative_improvement_error(df['retro_zenith_sys_width_track'],df['retro_zenith_nominal_width_track'],df['retro_zenith_sys_width_track_error'],df['retro_zenith_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        ax[1].errorbar(x - bar_width/2, (1- df['retro_energy_sys_width_track']/df['retro_energy_nominal_width_track'])*100, relative_improvement_error(df['retro_energy_sys_width_track'],df['retro_energy_nominal_width_track'],df['retro_energy_sys_width_track_error'],df['retro_energy_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        
        ax[2].errorbar(x - (3/2)*bar_width, (1 - df['dynedge_angular_res_sys_width_track']/df['dynedge_angular_res_nominal_width_track'])*100, relative_improvement_error(df['dynedge_angular_res_sys_width_track'],df['dynedge_angular_res_nominal_width_track'],df['dynedge_angular_res_sys_width_track_error'],df['dynedge_angular_res_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color =  gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[2].errorbar(x - bar_width/2, (1 - df['retro_angular_res_sys_width_track']/df['retro_angular_res_nominal_width_track'])*100, relative_improvement_error(df['retro_angular_res_sys_width_track'],df['retro_angular_res_nominal_width_track'],df['retro_angular_res_sys_width_track_error'],df['retro_angular_res_nominal_width_track_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        
        ax[0].errorbar(x + bar_width/2, (1- df['dynedge_zenith_sys_width_cascade']/df['dynedge_zenith_nominal_width_cascade'])*100,relative_improvement_error(df['dynedge_zenith_sys_width_cascade'],df['dynedge_zenith_nominal_width_cascade'],df['dynedge_zenith_sys_width_cascade_error'],df['dynedge_zenith_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[1].errorbar(x + bar_width/2, (1- df['dynedge_energy_sys_width_cascade']/df['dynedge_energy_nominal_width_cascade'])*100,relative_improvement_error(df['dynedge_energy_sys_width_cascade'],df['dynedge_energy_nominal_width_cascade'],df['dynedge_energy_sys_width_cascade_error'],df['dynedge_energy_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[0].errorbar(x + (3/2)*bar_width, (1- df['retro_zenith_sys_width_cascade']/df['retro_zenith_nominal_width_cascade'])*100,relative_improvement_error(df['retro_zenith_sys_width_cascade'],df['retro_zenith_nominal_width_cascade'],df['retro_zenith_sys_width_cascade_error'],df['retro_zenith_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')
        ax[1].errorbar(x + (3/2)*bar_width, (1- df['retro_energy_sys_width_cascade']/df['retro_energy_nominal_width_cascade'])*100,relative_improvement_error(df['retro_energy_sys_width_cascade'],df['retro_energy_nominal_width_cascade'],df['retro_energy_sys_width_cascade_error'],df['retro_energy_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')
        
        ax[2].errorbar(x + bar_width/2, (1 - df['dynedge_angular_res_sys_width_cascade']/df['dynedge_angular_res_nominal_width_cascade'])*100,relative_improvement_error(df['dynedge_angular_res_sys_width_cascade'],df['dynedge_angular_res_nominal_width_cascade'],df['dynedge_angular_res_sys_width_cascade_error'],df['dynedge_angular_res_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[2].errorbar(x + (3/2)*bar_width, (1 - df['retro_angular_res_sys_width_cascade']/df['retro_angular_res_nominal_width_cascade'])*100,relative_improvement_error(df['retro_angular_res_sys_width_cascade'],df['retro_angular_res_nominal_width_cascade'],df['retro_angular_res_sys_width_cascade_error'],df['retro_angular_res_nominal_width_cascade_error'])*100,fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')

    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].spines['bottom'].set_position(('data',0))
    ax[1].spines['right'].set_color('none')
    ax[1].spines['bottom'].set_position(('data',0))
    ax[1].spines['top'].set_color('none')
    ax[2].spines['right'].set_color('none')
    ax[2].spines['top'].set_color('none')
    
    #fig.text(0.04, 0.5, 'Resolution Improvement [%]', va='center', rotation='vertical', size = 12)
    ax[0].set_ylabel('Zenith Resolution Change (%)', size = 12)
    ax[1].set_ylabel('Energy Resolution Change (%)', size = 12)
    ax[2].set_ylabel('Direction Resolution Change (%)', size = 12)

    ylbl0 = ax[0].yaxis.get_label()
    ylbl2 = ax[2].yaxis.get_label()
    ylbl_target = ax[1].yaxis.get_label()
    ax[0].yaxis.set_label_coords(-0.07,ylbl0.get_position()[1])
    ax[2].yaxis.set_label_coords(-0.07,ylbl_target.get_position()[1])
    ax[1].yaxis.set_label_coords(-0.07,ylbl_target.get_position()[1])
    
    ax[1].set_xticks(x)
    ax[0].set_xticks(x)
    ax[2].set_xticks(x)
    new_ax = ax[2].twiny()
    new_ax.set_xticks(x)
    new_ax.set_xlim(ax[2].get_xlim())
    new_ax.xaxis.set_ticks_position('bottom')
    new_ax.tick_params(top=False,labeltop=False,labelbottom=False, bottom = True)

    new_ax.spines['left'].set_color('none')
    new_ax.spines['top'].set_color('none')
    new_ax.spines['right'].set_color('none')
    new_ax.spines['bottom'].set_position(('data',0))

    ax[2].set_xticklabels(set_values, rotation = 90, fontsize = 8)
    ax[1].grid()
    ax[1].yaxis.grid(False)
    ax[0].grid()
    ax[0].yaxis.grid(False)
    ax[2].grid()
    ax[2].yaxis.grid(False)
    #ax[1].legend()
    ax[0].legend(bbox_to_anchor=(-0.05,1.15), loc="upper left", ncol = 4)
    #fig.legend(ncol = 4, loc = 'upper center')
    ax[0].tick_params(bottom=True,labelbottom=False)
    ax[1].tick_params(bottom=True,labelbottom=False)
    #ax[2].set_xlabel('Systematic Set', size = 13)
    trans = ax[2].get_xaxis_transform()
    ax[2].annotate(text = 'DOM Efficiency', xy = (4.5,-0.42), xytext = (4.5,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 4 , lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    ax[2].annotate(text = 'Mixed Profiles (DE, p0, p1)', xy = (29,-0.42), xytext = (29,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 13, lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    ax[2].annotate(text = 'Bulk Ice (Scat., Att.)', xy = (52.5,-0.42), xytext = (52.5 + 2,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 4, lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    #transFigure = fig.transFigure.inverted()
    #coord1 = transFigure.transform(ax[0].transData.transform([-(10)*bar_width,4]))
    #coord2 = transFigure.transform(ax[2].transData.transform([-(10)*bar_width,-6]))
    #line1 = mpl.lines.Line2D((coord2[0],coord1[0]),(coord2[1],coord1[1]),
    #                           transform=fig.transFigure,
    #                           ls='--',
    #                           color='grey')
    #fig.lines.extend([line1])
    plt.subplots_adjust( 
                    wspace=0.05, 
                    hspace=0.05)
    resolution_filename = generate_file_name(plot_type = 'resolution')
    #ax[2].set_xlim(x[1] - x[1]*(1/2) , x[-1] + x[1]*(1/2))
    #new_ax.set_xlim(ax[2].get_xlim())
    plt.tight_layout() 
    print(resolution_filename)
    fig.savefig('plots/bar_' + resolution_filename)
    

    ### BIAS BAR PLOT
    fig, ax = plt.subplots(3,1, figsize = (8,10), sharex=  True)
    bar_width = 0.15*3
    fmt = '.'
    capsize = 4
    alpha_track_gnn = 1
    alpha_cascade_gnn = 0.5
    alpha_track_retro = 1
    alpha_cascade_retro = 0.5

    gnn_error_color_track = 'tab:blue'
    gnn_error_color_cascade = 'tab:blue'

    retro_error_color_track = 'tab:orange'
    retro_error_color_cascade = 'tab:orange'
    errorbars = True
    x = np.arange(0,len(df)*3,3)

    ax[0].bar(x - (3/2)*bar_width, df['dynedge_zenith_sys_bias_track'] - df['dynedge_zenith_nominal_bias_track'],bar_width,  label = 'GraphNet Track', color = 'tab:blue', alpha = 1)
    ax[1].bar(x - (3/2)*bar_width , df['dynedge_energy_sys_bias_track'] - df['dynedge_energy_nominal_bias_track'],bar_width, color = 'tab:blue', alpha = 1)
    ax[0].bar(x - bar_width/2, df['retro_zenith_sys_bias_track'] - df['retro_zenith_nominal_bias_track'],bar_width, label = 'Retro Track ', color = 'tab:orange', alpha = 1)
    ax[1].bar(x - bar_width/2, df['retro_energy_sys_bias_track'] - df['retro_energy_nominal_bias_track'], bar_width, color = 'tab:orange', alpha = 1)
    
    ax[2].bar(x - (3/2)*bar_width, df['dynedge_azimuth_sys_bias_track'] - df['dynedge_azimuth_nominal_bias_track'], bar_width, color = 'tab:blue', alpha = 1)
    ax[2].bar(x - bar_width/2, df['retro_azimuth_sys_bias_track'] - df['retro_azimuth_nominal_bias_track'], bar_width, color = 'tab:orange', alpha = 1)
    
    ax[0].bar(x + bar_width/2, df['dynedge_zenith_sys_bias_cascade'] - df['dynedge_zenith_nominal_bias_cascade'],bar_width, label = 'GraphNet Cascade', color = 'tab:blue', alpha = 0.5)
    ax[1].bar(x + bar_width/2, df['dynedge_energy_sys_bias_cascade'] - df['dynedge_energy_nominal_bias_cascade'],bar_width, color = 'tab:blue', alpha = 0.5)
    ax[0].bar(x + (3/2)*bar_width, df['retro_zenith_sys_bias_cascade'] - df['retro_zenith_nominal_bias_cascade'],bar_width, label = ' Retro Cascade ', color = 'tab:orange', alpha = 0.5)
    ax[1].bar(x + (3/2)*bar_width, df['retro_energy_sys_bias_cascade'] - df['retro_energy_nominal_bias_cascade'],bar_width, color = 'tab:orange', alpha = 0.5)
    
    ax[2].bar(x + bar_width/2,  df['dynedge_azimuth_sys_bias_cascade'] - df['dynedge_azimuth_nominal_bias_cascade'],bar_width, color = 'tab:blue', alpha = 0.5)
    ax[2].bar(x + (3/2)*bar_width, df['retro_azimuth_sys_bias_cascade'] - df['retro_azimuth_nominal_bias_cascade'], bar_width, color = 'tab:orange', alpha = 0.5)

    if errorbars == True:
        ax[0].errorbar(x - (3/2)*bar_width, df['dynedge_zenith_sys_bias_track'] - df['dynedge_zenith_nominal_bias_track'], bias_variation_error(df['dynedge_zenith_sys_bias_track'],df['dynedge_zenith_nominal_bias_track'],df['dynedge_zenith_sys_bias_track_error'],df['dynedge_zenith_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color = gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[1].errorbar(x - (3/2)*bar_width , df['dynedge_energy_sys_bias_track'] - df['dynedge_energy_nominal_bias_track'],bias_variation_error(df['dynedge_energy_sys_bias_track'],df['dynedge_energy_nominal_bias_track'],df['dynedge_energy_sys_bias_track_error'],df['dynedge_energy_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color = gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[0].errorbar(x - bar_width/2, df['retro_zenith_sys_bias_track'] - df['retro_zenith_nominal_bias_track'],bias_variation_error(df['retro_zenith_sys_bias_track'],df['retro_zenith_nominal_bias_track'],df['retro_zenith_sys_bias_track_error'],df['retro_zenith_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        ax[1].errorbar(x - bar_width/2, df['retro_energy_sys_bias_track'] - df['retro_energy_nominal_bias_track'], bias_variation_error(df['retro_energy_sys_bias_track'],df['retro_energy_nominal_bias_track'],df['retro_energy_sys_bias_track_error'],df['retro_energy_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        
        ax[2].errorbar(x - (3/2)*bar_width, df['dynedge_azimuth_sys_bias_track'] - df['dynedge_azimuth_nominal_bias_track'], bias_variation_error(df['dynedge_azimuth_sys_bias_track'],df['dynedge_azimuth_nominal_bias_track'],df['dynedge_azimuth_sys_bias_track_error'],df['dynedge_azimuth_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color =  gnn_error_color_track, alpha = alpha_track_gnn, ecolor = 'black')
        ax[2].errorbar(x - bar_width/2, df['retro_azimuth_sys_bias_track'] - df['retro_azimuth_nominal_bias_track'], bias_variation_error(df['retro_azimuth_sys_bias_track'],df['retro_azimuth_nominal_bias_track'],df['retro_azimuth_sys_bias_track_error'],df['retro_azimuth_nominal_bias_track_error']),fmt = fmt,capsize = capsize, color = retro_error_color_track, alpha = alpha_track_retro, ecolor = 'black')
        
        ax[0].errorbar(x + bar_width/2, df['dynedge_zenith_sys_bias_cascade'] - df['dynedge_zenith_nominal_bias_cascade'],bias_variation_error(df['dynedge_zenith_sys_bias_cascade'],df['dynedge_zenith_nominal_bias_cascade'],df['dynedge_zenith_sys_bias_cascade_error'],df['dynedge_zenith_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[1].errorbar(x + bar_width/2, df['dynedge_energy_sys_bias_cascade'] - df['dynedge_energy_nominal_bias_cascade'],bias_variation_error(df['dynedge_energy_sys_bias_cascade'],df['dynedge_energy_nominal_bias_cascade'],df['dynedge_energy_sys_bias_cascade_error'],df['dynedge_energy_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[0].errorbar(x + (3/2)*bar_width, df['retro_zenith_sys_bias_cascade'] - df['retro_zenith_nominal_bias_cascade'],bias_variation_error(df['retro_zenith_sys_bias_cascade'],df['retro_zenith_nominal_bias_cascade'],df['retro_zenith_sys_bias_cascade_error'],df['retro_zenith_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')
        ax[1].errorbar(x + (3/2)*bar_width, df['retro_energy_sys_bias_cascade'] - df['retro_energy_nominal_bias_cascade'],bias_variation_error(df['retro_energy_sys_bias_cascade'],df['retro_energy_nominal_bias_cascade'],df['retro_energy_sys_bias_cascade_error'],df['retro_energy_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')
        
        ax[2].errorbar(x + bar_width/2, df['dynedge_azimuth_sys_bias_cascade'] - df['dynedge_azimuth_nominal_bias_cascade'],bias_variation_error(df['dynedge_azimuth_sys_bias_cascade'],df['dynedge_azimuth_nominal_bias_cascade'],df['dynedge_azimuth_sys_bias_cascade_error'],df['dynedge_azimuth_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = gnn_error_color_cascade, alpha = alpha_cascade_gnn, ecolor = 'black')
        ax[2].errorbar(x + (3/2)*bar_width, df['retro_azimuth_sys_bias_cascade'] - df['retro_azimuth_nominal_bias_cascade'],bias_variation_error(df['retro_azimuth_sys_bias_cascade'],df['retro_azimuth_nominal_bias_cascade'],df['retro_azimuth_sys_bias_cascade_error'],df['retro_azimuth_nominal_bias_cascade_error']),fmt = fmt,capsize = capsize, color = retro_error_color_cascade, alpha = alpha_cascade_retro, ecolor = 'black')


    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')
    ax[0].spines['bottom'].set_position(('data',0))
    ax[1].spines['right'].set_color('none')
    ax[1].spines['bottom'].set_position(('data',0))
    ax[1].spines['top'].set_color('none')
    ax[2].spines['right'].set_color('none')
    ax[2].spines['top'].set_color('none')
    
    
    #ax[1].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    #ax[0].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)
    #ax[2].plot(np.arange(0,len(df)), np.repeat(0, len(df)), color = 'black', lw = 4)

    #ax[1].set_ylim([-12,12])
    #ax[0].set_ylim([-12,12])
    #ax[2].set_ylim([-12,12])
    
    #fig.text(0.04, 0.5, 'Resolution Improvement [%]', va='center', rotation='vertical', size = 12)
    ax[0].set_ylabel('Zenith Bias Change (Deg.)', size = 12)
    ax[1].set_ylabel('Energy Bias Change (%)', size = 12)
    ax[2].set_ylabel('Azimuth Bias Change (Deg.)', size = 12)

    ylbl0 = ax[0].yaxis.get_label()
    ylbl2 = ax[2].yaxis.get_label()
    ylbl_target = ax[1].yaxis.get_label()
    ax[0].yaxis.set_label_coords(-0.07,ylbl0.get_position()[1])
    ax[2].yaxis.set_label_coords(-0.07,ylbl_target.get_position()[1])
    ax[1].yaxis.set_label_coords(-0.07,ylbl_target.get_position()[1])
    
    ax[1].set_xticks(x)
    ax[0].set_xticks(x)
    ax[2].set_xticks(x)
    new_ax = ax[2].twiny()
    new_ax.set_xticks(x)
    new_ax.set_xlim(ax[2].get_xlim())
    new_ax.xaxis.set_ticks_position('bottom')
    new_ax.tick_params(top=False,labeltop=False,labelbottom=False, bottom = True)

    new_ax.spines['left'].set_color('none')
    new_ax.spines['top'].set_color('none')
    new_ax.spines['right'].set_color('none')
    new_ax.spines['bottom'].set_position(('data',0))

    ax[2].set_xticklabels(set_values, rotation = 90, fontsize = 8)
    ax[1].grid()
    ax[1].yaxis.grid(False)
    ax[0].grid()
    ax[0].yaxis.grid(False)
    ax[2].grid()
    ax[2].yaxis.grid(False)
    #ax[1].legend()
    ax[0].legend(bbox_to_anchor=(-0.05,1.15), loc="upper left", ncol = 4)
    #fig.legend(ncol = 4, loc = 'upper center')
    ax[0].tick_params(bottom=True,labelbottom=False)
    ax[1].tick_params(bottom=True,labelbottom=False)
    #ax[2].set_xlabel('Systematic Set', size = 13)
    trans = ax[2].get_xaxis_transform()
    ax[2].annotate(text = 'DOM Efficiency', xy = (7,-0.42), xytext = (7,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 4 , lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    ax[2].annotate(text = 'Mixed Profiles (DE, p0, p1)', xy = (31.5,-0.42), xytext = (31.5,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 13.4, lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    ax[2].annotate(text = 'Bulk Ice (Scat., Att.)', xy = (56,-0.42), xytext = (56,-0.49), xycoords=trans ,color="tab:blue", horizontalalignment="center", verticalalignment="center",
    arrowprops=dict(arrowstyle='-[, widthB = 4, lengthB=0.4', connectionstyle="bar, angle = 0, fraction = 0",color="k"))

    #fig.suptitle('Robustness of Resolution', size = 8)    
    #ax[0].set_title('Zenith')
    #ax[1].set_title('Energy')
    #ax[2].set_title('Direction')
    plt.subplots_adjust( 
                    wspace=0.05, 
                    hspace=0.05)
    bias_filename = generate_file_name(plot_type = 'bias')
    #ax[2].set_xlim(x[1] - x[1]*(1/2) , x[-1] + x[1]*(1/2))
    #new_ax.set_xlim(ax[2].get_xlim())
    plt.tight_layout() 
    print(resolution_filename)
    move_old_plots(bias_filename, resolution_filename)
    fig.savefig('plots/bar_' + bias_filename)

    #### CLASSIFICATION
    fig, ax = plt.subplots(2,1, figsize = (8,8))
    
    ax[0].errorbar(np.arange(0,len(df)), (-1 + df['dynedge_neutrino_sys_auc']/df['dynedge_neutrino_nominal_auc'])*100,marker='o', markeredgecolor='black', label = 'DynEdge')
    ax[1].errorbar(np.arange(0,len(df)), (-1 + df['dynedge_track_sys_auc']/df['dynedge_track_nominal_auc'])*100,marker='o', markeredgecolor='black', label = 'DynEdge')
    ax[0].errorbar(np.arange(0,len(df)), (-1 + df['retro_neutrino_sys_auc']/df['retro_neutrino_nominal_auc'])*100,marker='o', markeredgecolor='black', label = 'BDT')
    ax[1].errorbar(np.arange(0,len(df)), (-1 + df['retro_track_sys_auc']/df['retro_track_nominal_auc'])*100,marker='o', markeredgecolor='black', label = 'BDT')
    
 

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
    ax[1].set_xticklabels(set_values, rotation = 25)
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
    data_folder = '/mnt/scratch/rasmus_orsoe/paper_data/data_with_overlap_labels'
    #count_events_in_systematic_sets(data_folder)
    make_robustness_plots(data_folder, from_csv = False, overlap_only= False, relative_overlap = True)
