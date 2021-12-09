import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
from time import strftime,gmtime
import os

from pathlib import Path
from scipy import stats
from scipy import stats
from copy import deepcopy


def AddSignature(db, df):
    events  = df['event_no']
    with sqlite3.connect(db) as con:
        query = 'select event_no, pid, interaction_type from truth where event_no in %s'%str(tuple(events))
        data = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
        
    df = df.sort_values('event_no').reset_index(drop = 'True')
    df['signature'] =  int((abs(data['pid']) == 14) & (data['interaction_type'] == 1))
    return df


def CalculateWidth(bias_tmp):
    return (np.percentile(bias_tmp,84) - np.percentile(bias_tmp,16))/2
    #return (np.percentile(bias_tmp,75) - np.percentile(bias_tmp,25))/1.365

def gauss_pdf(mean, std, x):
    pdf =  1/(std*np.sqrt(2*np.pi)) * np.exp(-(1/2)*((x-mean)/std)**2)
    return (pdf).reset_index(drop = True)

def empirical_pdf(x,diff):
    dist = getattr(stats, 'norm')
    parameters = dist.fit(diff)
    pdf = gauss_pdf(parameters[0],parameters[1],diff)[x]
    #print(pdf)
    return pdf

def CalculateWidthError(diff):
    N = len(diff)
    x_16 = abs(diff-np.percentile(diff,16,interpolation='nearest')).argmin() #int(0.16*N)
    x_84 = abs(diff-np.percentile(diff,84,interpolation='nearest')).argmin() #int(0.84*N)
    fe_16 = sum(diff <= diff[x_16])/N
    fe_84 = sum(diff <= diff[x_84])/N
    n_16 = sum(diff <= diff[x_16])
    n_84 = sum(diff <= diff[x_84]) 
    #error_width = np.sqrt((0.16*(1-0.16)/N)*(1/fe_16**2 + 1/fe_84**2))*(1/2)
    #n,bins,_ = plt.hist(diff, bins = 30)
    #plt.close()
    if len(diff)>0:
        error_width = np.sqrt((1/empirical_pdf(x_84, diff)**2)*(0.84*(1-0.84)/N) + (1/empirical_pdf(x_16, diff)**2)*(0.16*(1-0.16)/N))*(1/2)
    else:
        error_width = np.nan
    return error_width
   
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
    print(data.columns)
    data = convert_to_unit_vectors(data, post_fix)
    dotprod = (data['x']*data['x' + post_fix].values + data['y']*data['y'+ post_fix].values + data['z']*data['z'+ post_fix].values)
    norm_data = np.sqrt(data['x'+ post_fix]**2 + data['y'+ post_fix]**2 + data['z'+ post_fix]**2).values
    norm_truth = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2).values

    cos_angle = dotprod/(norm_data*norm_truth)

    return np.arccos(cos_angle).values*(360/(2*np.pi))


def calculate_xyz_difference(data,is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    #if is_retro == False:
    diff = np.sqrt((data['position_x'] - data['position_x%s'%post_fix])**2 + (data['position_y'] - data['position_y%s'%post_fix])**2 + (data['position_z'] - data['position_z%s'%post_fix])**2)
    return diff

def ExtractStatistics(data_raw,keys, key_bins, is_retro):
    data_raw = data_raw.sort_values('event_no').reset_index(drop = 'True')
    pids = pd.unique(abs(data_raw['pid']))
    interaction_types = data_raw['interaction_type'].unique()
    biases = {}
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    for key in keys:
        data = deepcopy(data_raw)
        print(key)
        biases[key] = {}
        if key not in ['energy', 'angular_res', 'XYZ', 'interaction_time']:
            data[key] = data[key]*(360/(2*np.pi))
            data[key + post_fix] = data[key + post_fix]*(360/(2*np.pi))
        if key == 'angular_res':
            data[key] = calculate_angular_difference(data, is_retro)
        if key == 'XYZ':
            data[key] = calculate_xyz_difference(data,is_retro)
        for pid in pids:
            biases[key][str(pid)] = {}
            data_pid_indexed = data.loc[abs(data['pid']) == pid,:].reset_index(drop = True)
            for interaction_type in interaction_types:
                biases[key][str(pid)][str(interaction_type)] = {'mean':         [],
                                                                '16th':         [],
                                                                '50th':         [],
                                                                '84th':         [],
                                                                'count':        [],
                                                                'width':        [],
                                                                'width_error':  [],
                                                                'predictions' : [],
                                                                'bias': []}
                data_interaction_indexed = data_pid_indexed.loc[data_pid_indexed['interaction_type'] == interaction_type,:]
                if len(data_interaction_indexed) > 0:
                    if key not in ['angular_res', 'XYZ']:
                        biases[key][str(pid)][str(interaction_type)]['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
                    if key == 'angular_res':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = data_interaction_indexed['angular_res']
                    if key == 'energy':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
                    if key == 'zenith' or key == 'interaction_time':
                        biases[key][str(pid)][str(interaction_type)]['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
                bins = key_bins['energy']

                for i in range(1,(len(bins))):
                    bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
                    data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
                    
                    if key == 'energy':
                        bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/10**(data_interaction_indexed_sliced[key]))*100
                        bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
                    if key in ['zenith', 'azimuth', 'interaction_time']:
                        bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
                        if key == 'azimuth':
                            bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                            bias_tmp[bias_tmp<= -180] = -(bias_tmp[bias_tmp<= -180] + 360)
                    if key in ['angular_res', 'XYZ']:
                        bias_tmp = data_interaction_indexed_sliced[key]
                    if len(data_interaction_indexed_sliced)>0:
                        biases[key][str(pid)][str(interaction_type)]['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                        
                        #biases[key][str(pid)][str(interaction_type)]['count'].append(len(bias_tmp))
                        #biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                        #biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                        if key == 'energy':
                            biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp_percent))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp_percent))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp_percent,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp_percent,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp_percent,84))
                        else:
                            biases[key][str(pid)][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                            biases[key][str(pid)][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                            biases[key][str(pid)][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                            biases[key][str(pid)][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                            biases[key][str(pid)][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
        
        biases[key]['all_pid'] = {}
        for interaction_type in interaction_types:
            biases[key]['all_pid'][str(interaction_type)] = {'mean':         [],
                                                            '16th':         [],
                                                            '50th':         [],
                                                            '84th':         [],
                                                            'count':        [],
                                                            'width':        [],
                                                            'width_error':  [],
                                                            'predictions': []}
            data_interaction_indexed = data.loc[data['interaction_type'] == interaction_type,:]
            if len(data_interaction_indexed) > 0:
                if key not in ['angular_res', 'XYZ']: 
                    biases[key]['all_pid'][str(interaction_type)]['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
                if key in ['angular_res', 'XYZ']:
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = data_interaction_indexed[key]
                elif key == 'energy':
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
                else:
                    biases[key]['all_pid'][str(interaction_type)]['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
            bins = key_bins['energy']
            for i in range(1,(len(bins))):
                bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
                data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
                
                if key == 'energy':
                    bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/(10**(data_interaction_indexed_sliced[key])))*100
                    bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
                elif key not in ['angular_res', 'XYZ']:
                    bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
                else:
                    bias_tmp = data_interaction_indexed_sliced[key]

                if key == 'azimuth':
                    bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                    bias_tmp[bias_tmp<= -180] = (bias_tmp[bias_tmp<= -180] + 360)
                    if np.max(bias_tmp) > 180:
                        print(np.max(bias_tmp))
                if len(data_interaction_indexed_sliced)>0:
                    biases[key]['all_pid'][str(interaction_type)]['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                    biases[key]['all_pid'][str(interaction_type)]['count'].append(len(bias_tmp))
                    if key == 'energy':
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp_percent))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp_percent))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp_percent,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp_percent,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp_percent,84))
                    else:
                        biases[key]['all_pid'][str(interaction_type)]['width'].append(CalculateWidth(bias_tmp))
                        biases[key]['all_pid'][str(interaction_type)]['width_error'].append(CalculateWidthError(bias_tmp))
                        biases[key]['all_pid'][str(interaction_type)]['16th'].append(np.percentile(bias_tmp,16))
                        biases[key]['all_pid'][str(interaction_type)]['50th'].append(np.percentile(bias_tmp,50))
                        biases[key]['all_pid'][str(interaction_type)]['84th'].append(np.percentile(bias_tmp,84))
        
        biases[key]['cascade'] = {}
        biases[key]['cascade']                       = {'mean':         [],
                                                        '16th':         [],
                                                        '50th':         [],
                                                        '84th':         [],
                                                        'count':        [],
                                                        'width':        [],
                                                        'width_error':  [],
                                                        'predictions': []}
        data_interaction_indexed = data.loc[~((data['pid'] == 14.0) & (data['interaction_type'] == 1.0)) ,:]
        if len(data_interaction_indexed) > 0:
            if key not in ['angular_res', 'XYZ']: 
                biases[key]['cascade']['predictions'] = data_interaction_indexed[key + post_fix].values.ravel()
            if key in ['angular_res', 'XYZ']:
                biases[key]['cascade']['bias'] = data_interaction_indexed[key]
            if key == 'energy':
                biases[key]['cascade']['bias'] = ((10**(data_interaction_indexed[key + post_fix])- 10**(data_interaction_indexed[key]))/10**(data_interaction_indexed[key]))
            if key not in ['angular_res', 'XYZ']:
                biases[key]['cascade']['bias'] = (data_interaction_indexed[key +  post_fix] - data_interaction_indexed[key]).values.ravel()
        bins = key_bins['energy']
        for i in range(1,(len(bins))):
            bin_index  = (data_interaction_indexed['energy'] > bins[i-1]) & (data_interaction_indexed['energy'] < bins[i])
            data_interaction_indexed_sliced = data_interaction_indexed.loc[bin_index,:].sort_values('%s'%key).reset_index(drop  = True) 
            
            if key == 'energy':
                bias_tmp_percent = ((10**(data_interaction_indexed_sliced[key + post_fix])- 10**(data_interaction_indexed_sliced[key]))/(10**(data_interaction_indexed_sliced[key])))*100
                bias_tmp = data_interaction_indexed_sliced[key +  post_fix] - data_interaction_indexed_sliced[key]
            if key not in ['angular_res', 'XYZ']:
                bias_tmp = data_interaction_indexed_sliced[key +  post_fix]- data_interaction_indexed_sliced[key]
            else:
                bias_tmp = data_interaction_indexed_sliced[key]
            if key == 'azimuth':
                bias_tmp[bias_tmp>= 180] = 360 - bias_tmp[bias_tmp>= 180]
                bias_tmp[bias_tmp<= -180] = (bias_tmp[bias_tmp<= -180] + 360)
                if np.max(bias_tmp) > 180:
                    print(np.max(bias_tmp))
            if len(data_interaction_indexed_sliced)>0:
                biases[key]['cascade']['mean'].append(np.mean(data_interaction_indexed_sliced['energy']))
                biases[key]['cascade']['count'].append(len(bias_tmp))
                if key == 'energy':
                    biases[key]['cascade']['width'].append(CalculateWidth(bias_tmp_percent))
                    biases[key]['cascade']['width_error'].append(CalculateWidthError(bias_tmp_percent))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp_percent,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp_percent,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp_percent,84))
                else:
                    biases[key]['cascade']['width'].append(CalculateWidth(bias_tmp))
                    biases[key]['cascade']['width_error'].append(CalculateWidthError(bias_tmp))
                    biases[key]['cascade']['16th'].append(np.percentile(bias_tmp,16))
                    biases[key]['cascade']['50th'].append(np.percentile(bias_tmp,50))
                    biases[key]['cascade']['84th'].append(np.percentile(bias_tmp,84))
    return biases

    
def CalculateStatistics(data,keys, key_bins,include_retro = False):
    biases = {'dynedge': ExtractStatistics(data, keys, key_bins, is_retro = False)}
    if include_retro:
        biases['retro'] = ExtractStatistics(data, keys, key_bins, is_retro = True)
    return biases
    

def CalculateRelativeImprovementError(relimp, w1, w1_sigma, w2, w2_sigma):
    sigma = np.sqrt((np.array(w1_sigma)/np.array(w1))**2 + (np.array(w2_sigma)/np.array(w2))**2)
    return sigma

def MakeSummaryWidthPlot(key_limits, biases, include_retro, track_cascade = False):
    key_limits = key_limits['width']
    if track_cascade == False:
        for key in biases['dynedge'].keys():
            fig = plt.figure()
            plt.hist(biases['dynedge'][key]['all_pid']['1.0']['predictions'], histtype = 'step', label = 'dynedge')
            plt.hist(biases['retro'][key]['all_pid']['1.0']['predictions'], histtype = 'step', label = 'retro')
            plt.legend()
            fig.savefig('test_hist.png')
            fig = plt.figure()
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
            pid_count = 0
            pid = 'all_pid'
            interaction_type = str(1.0)
            plot_data = biases['dynedge'][key][pid][interaction_type]
            if include_retro:
                plot_data_retro = biases['retro'][key][pid][interaction_type]
            if len(plot_data['mean']) != 0:
                ax3 = ax1.twinx()
                ax3.bar(x = (plot_data['mean']), height = plot_data['count'], 
                        alpha = 0.3, 
                        color = 'grey',
                        align = 'center',
                        width = 0.25)
                ax1.errorbar(plot_data['mean'],plot_data['width'],plot_data['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, label = 'dynedge')
                if include_retro:
                    ax1.errorbar(plot_data_retro['mean'],plot_data_retro['width'],plot_data_retro['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, label = 'RetroReco')
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(plot_data['mean'], np.repeat(0, len(plot_data['mean'])), color = 'black', lw = 2)
                #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
                if include_retro:
                    ax2.errorbar(plot_data['mean'],1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']),CalculateRelativeImprovementError(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']), plot_data['width'], plot_data['width_error'], plot_data_retro['width'], plot_data_retro['width_error']),marker='o', capsize = 10,markeredgecolor='black')

                #plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis='x', labelsize=10)
                ax1.tick_params(axis='y', labelsize=10)
                ax1.set_xlim(key_limits[key]['x'])
                ax2.set_xlim(key_limits[key]['x'])
                ax2.set_ylim([-0.1,0.55])
                ax1.legend()
                if key == 'energy':
                    unit_tag = '[%]'
                else:
                    unit_tag = '[deg.]'
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key, unit_tag), size = 15)
                ax2.set_xlabel('$Energy_{log10}$ [GeV]', size = 15)
                ax2.set_ylabel('Rel. Impro.', size = 15)  
                
                fig.suptitle('%s CC'%key, size = 20)
                fig.savefig('performance_%s.png'%key)
    else:
        for key in biases['dynedge'].keys():
            fig = plt.figure()
            ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
            ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
            pid_count = 0
            pid = 'track'
            interaction_type = str(1.0)
            plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
            plot_data_cascade = biases['dynedge'][key]['cascade']
            if include_retro:
                plot_data_retro_track = biases['retro'][key][str(14.0)][str(1.0)]
                plot_data_retro_cascade = biases['retro'][key]['cascade']
            if len(plot_data_track['mean']) != 0:
                ax3 = ax1.twinx()
                #ax3.bar(x = plot_data_track['mean'], height = plot_data_track['count'], 
                #        alpha = 0.3, 
                #        color = 'grey',
                #        align = 'center',
                #        width = 0.25)
                print(len(plot_data_track['width']))
                print(len(plot_data_track['mean']))
                ax1.errorbar(plot_data_track['mean'],plot_data_track['width'],plot_data_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'blue', label = 'GCN-all Track')
                ax1.errorbar(plot_data_cascade['mean'],plot_data_cascade['width'],plot_data_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkblue', label = 'GCN-all Cascade')
                if include_retro:
                    ax1.errorbar(plot_data_retro_track['mean'],plot_data_retro_track['width'],plot_data_retro_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'orange', label = 'RetroReco Track')
                    ax1.errorbar(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],plot_data_retro_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkorange' , label = 'RetroReco Cascade')
                labels = [item.get_text() for item in ax1.get_xticklabels()]
                empty_string_labels = ['']*len(labels)
                ax1.set_xticklabels(empty_string_labels)
                ax1.grid()
                ax2.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'black', lw = 2)
                #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
                if include_retro:
                    ax2.errorbar(plot_data_track['mean'],1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'limegreen', label = 'track',linestyle='dotted')
                    ax2.errorbar(plot_data_cascade['mean'],1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'springgreen', label = 'cascade',linestyle='solid')
                    ax2.legend()
                #plt.title('$\\nu_{v,u,e}$', size = 20)
                ax1.tick_params(axis='x', labelsize=10)
                ax1.tick_params(axis='y', labelsize=10)
                ax1.set_xlim(key_limits[key]['x'])
                ax2.set_xlim(key_limits[key]['x'])
                ax2.set_ylim([-0.40,0.40])
                ax1.legend()
                if key == 'energy':
                    unit_tag = '[%]'
                else:
                    unit_tag = '[deg.]'
                plt.tick_params(right=False,labelright=False)
                ax1.set_ylabel('%s Resolution %s'%(key, unit_tag), size = 15)
                ax2.set_xlabel('$Energy_{log10}$ [GeV]', size = 15)
                ax2.set_ylabel('Rel. Impro.', size = 15)  
                
                fig.suptitle('%s Performance'%key, size = 20)
                fig.savefig('performance_track_cascade_outline_presentation_%s.png'%key)

    return fig

def transform_energy(data):
    data['energy'] = np.log10(data['energy'])
    data['energy_pred'] = np.log10(data['energy_pred'])
    data['energy_retro'] = np.log10(data['energy_retro'])
    return data

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data

def make_resolution_plots(targets, plot_config, include_retro, track_cascade = False):
    data = pd.read_csv(plot_config['data'])
    data = transform_energy(data)
    data = remove_muons(data)
    key_limits = plot_config['width']
    key_bins = plot_config['key_bins']
    biases = CalculateStatistics(data,targets, key_bins,include_retro = True)
    for key in biases['dynedge'].keys():
        fig = plt.figure()
        ax1 = plt.subplot2grid((6, 6), (0, 0), colspan = 6, rowspan= 4)
        ax2 = plt.subplot2grid((6, 6), (4, 0), colspan = 6, rowspan= 2)
        pid_count = 0
        pid = 'track'
        interaction_type = str(1.0)
        plot_data_track = biases['dynedge'][key][str(14.0)][str(1.0)]
        plot_data_cascade = biases['dynedge'][key]['cascade']
        if include_retro:
            plot_data_retro_track = biases['retro'][key][str(14.0)][str(1.0)]
            plot_data_retro_cascade = biases['retro'][key]['cascade']
        if len(plot_data_track['mean']) != 0:
            ax3 = ax1.twinx()
            #ax3.bar(x = plot_data_track['mean'], height = plot_data_track['count'], 
            #        alpha = 0.3, 
            #        color = 'grey',
            #        align = 'center',
            #        width = 0.25)
            print(len(plot_data_track['width']))
            print(len(plot_data_track['mean']))
            ax1.errorbar(plot_data_track['mean'],plot_data_track['width'],plot_data_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'blue', label = 'GCN-all Track')
            ax1.errorbar(plot_data_cascade['mean'],plot_data_cascade['width'],plot_data_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkblue', label = 'GCN-all Cascade')
            if include_retro:
                ax1.errorbar(plot_data_retro_track['mean'],plot_data_retro_track['width'],plot_data_retro_track['width_error'],linestyle='dotted',fmt = 'o',capsize = 10, color = 'orange', label = 'RetroReco Track')
                ax1.errorbar(plot_data_retro_cascade['mean'],plot_data_retro_cascade['width'],plot_data_retro_cascade['width_error'],linestyle='solid',fmt = 'o',capsize = 10, color = 'darkorange' , label = 'RetroReco Cascade')
            labels = [item.get_text() for item in ax1.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax1.set_xticklabels(empty_string_labels)
            ax1.grid()
            ax2.plot(plot_data_track['mean'], np.repeat(0, len(plot_data_track['mean'])), color = 'black', lw = 2)
            #rel_imp_error = abs(1 - np.array(plot_data['width'])/np.array(plot_data_retro['width']))*np.sqrt((np.array(plot_data_retro['width_error'])/np.array(plot_data_retro['width']))**2 + (np.array(plot_data['width_error'])/np.array(plot_data['width']))**2)
            if include_retro:
                ax2.errorbar(plot_data_track['mean'],1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_track['width'])/np.array(plot_data_retro_track['width']), plot_data_track['width'], plot_data_track['width_error'], plot_data_retro_track['width'], plot_data_retro_track['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'limegreen', label = 'track',linestyle='dotted')
                ax2.errorbar(plot_data_cascade['mean'],1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']),CalculateRelativeImprovementError(1 - np.array(plot_data_cascade['width'])/np.array(plot_data_retro_cascade['width']), plot_data_cascade['width'], plot_data_cascade['width_error'], plot_data_retro_cascade['width'], plot_data_retro_cascade['width_error']),marker='o', capsize = 10,markeredgecolor='black', color = 'springgreen', label = 'cascade',linestyle='solid')
                ax2.legend()
            #plt.title('$\\nu_{v,u,e}$', size = 20)
            ax1.tick_params(axis='x', labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)
            ax1.set_xlim(key_limits[key]['x'])
            ax2.set_xlim(key_limits[key]['x'])
            ax2.set_ylim([-0.40,0.40])
            ax1.legend()
            if key == 'energy':
                unit_tag = '[%]'
            else:
                unit_tag = '[deg.]'
            plt.tick_params(right=False,labelright=False)
            ax1.set_ylabel('%s Resolution %s'%(key, unit_tag), size = 15)
            ax2.set_xlabel('$Energy_{log10}$ [GeV]', size = 15)
            ax2.set_ylabel('Rel. Impro.', size = 15)  
            
            fig.suptitle('%s Performance'%key, size = 20)
            fig.savefig('performance_track_cascade_outline_presentation_%s.png'%key)

    return fig


width_limits = {'energy':{'x':[0,3], 'y':[-0.5,1.5]},
                'zenith': {'x':[0,3], 'y':[-100,100]},
                'azimuth': {'x':[0,3], 'y':[-100,100]},
                'XYZ': {'x':[0,3], 'y':[-100,100]},
                'angular_res': {'x':[0,3], 'y':[-100,100]}}

key_bins = { 'energy': np.arange(0, 3.25, 0.25),
                'zenith': np.arange(0, 180, 10),
                'azimuth': np.arange(0, 2*180, 20) }


plot_config = {'data': '/groups/hep/pcs557/github/gnn_paper_plot_code/data/0000/everything.csv',
                'width': width_limits,
                'key_bins': key_bins}
#['zenith', 'energy','azimuth' ,'angular_res', 'XYZ']
make_resolution_plots(targets = ['zenith', 'energy','azimuth' ,'angular_res', 'XYZ'], plot_config = plot_config, include_retro= True, track_cascade = True)
