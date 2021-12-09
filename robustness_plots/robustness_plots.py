import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from pandas.core.algorithms import diff

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

def calculate_width(residual):
    return (np.percentile(residual,84) - np.percentile(residual,16))/2

def calculate_residual(data, target, is_retro):
    if is_retro:
        post_fix = '_retro'
    else:
        post_fix = '_pred'
    if target == 'energy':
        residual = ((np.log10(data[target + post_fix]) - np.log10(data[target]))/np.log10(data[target]))*100
    if target in ['azimuth', 'zenith']:
        residual = (data[target + post_fix] - data[target])*(360/(2*np.pi))
    if target == 'angular_res':
        residual = calculate_angular_difference(data, is_retro)
    return residual


def generate_file_name(plot_type):
    now = datetime.now()
    return plot_type + '_' + '-'.join(str(now).split(' '))[0:16]+ '.png'

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

def read_csv_and_make_statistics(data_folder):
    folders = os.listdir(data_folder)
    statistics = {'systematic': [],
                  'retro_bias_zenith': [],
                  'retro_bias_energy': [],
                  'retro_bias_angular_res': [],
                  'retro_width_zenith': [],
                  'retro_width_energy': [],
                  'retro_width_angular_res': [],
                  'dynedge_bias_zenith': [],
                  'dynedge_bias_energy': [],
                  'dynedge_bias_angular_res': [],
                  'dynedge_width_zenith': [],
                  'dynedge_width_energy': [],
                  'dynedge_width_angular_res': []}
    count = len(folders)-1
    for folder in folders:
        print('Reading %s. %s folders left.'%(folder, count))
        data = pd.read_csv(data_folder + '/' + folder + '/' + 'everything.csv')
        data = remove_muons(data)
        statistics['systematic'].append(folder)
        statistics['retro_bias_zenith'].append(np.percentile(calculate_residual(data, 'zenith', is_retro = True), 50))
        statistics['retro_bias_energy'].append(np.percentile(calculate_residual(data, 'energy', is_retro = True), 50))
        statistics['retro_bias_angular_res'].append(np.percentile(calculate_residual(data, 'angular_res', is_retro = True), 50))
        statistics['retro_width_zenith'].append(calculate_width(calculate_residual(data, 'zenith', is_retro = True)))
        statistics['retro_width_energy'].append(calculate_width(calculate_residual(data, 'energy', is_retro = True)))
        statistics['retro_width_angular_res'].append(calculate_width(calculate_residual(data, 'angular_res', is_retro = True)))

        statistics['dynedge_bias_zenith'].append(np.percentile(calculate_residual(data, 'zenith', is_retro = False), 50))
        statistics['dynedge_bias_energy'].append(np.percentile(calculate_residual(data, 'energy', is_retro = False), 50))
        statistics['dynedge_bias_angular_res'].append(np.percentile(calculate_residual(data, 'angular_res', is_retro = False), 50))
        statistics['dynedge_width_zenith'].append(calculate_width(calculate_residual(data, 'zenith', is_retro = False)))
        statistics['dynedge_width_energy'].append(calculate_width(calculate_residual(data, 'energy', is_retro = False)))
        statistics['dynedge_width_angular_res'].append(calculate_width(calculate_residual(data, 'angular_res', is_retro = False)))
        count = count -1 
    df = pd.DataFrame(statistics)
    df.to_csv('robustness_statistics.csv')
    return df
def make_robustness_plots(data_folder, from_csv = False):
    if from_csv:
        df = read_csv_and_make_statistics(data_folder).sort_values('systematic').reset_index(drop = True)
    else:
        df = pd.read_csv('robustness_statistics.csv', dtype={'systematic': object}).sort_values('systematic').reset_index(drop = True)
    ### RESOLUTION
    fig, ax = plt.subplots(3,1,figsize=(11.69,8.27))
    
    ax[0].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_zenith']/df['dynedge_width_zenith'][0])*100,marker='o', markeredgecolor='black', label = 'GCN')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['dynedge_width_energy']/df['dynedge_width_energy'][0])*100,marker='o', markeredgecolor='black', label = 'GCN')
    ax[0].errorbar(np.arange(0,len(df)), (1- df['retro_width_zenith']/df['retro_width_zenith'][0])*100,marker='o', markeredgecolor='black', label = 'RetroReco')
    ax[1].errorbar(np.arange(0,len(df)), (1- df['retro_width_energy']/df['retro_width_energy'][0])*100,marker='o', markeredgecolor='black', label = 'RetroReco')
    
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['dynedge_width_angular_res']/df['dynedge_width_angular_res'][0])*100,marker='o', markeredgecolor='black', label = 'GCN')
    ax[2].errorbar(np.arange(0,len(df)), (1 - df['retro_width_angular_res']/df['retro_width_angular_res'][0])*100,marker='o', markeredgecolor='black', label = 'RetroReco')
    
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
    fig.text(0.04, 0.5, 'Variation w.r.t. Nominal [%]', va='center', rotation='vertical', size = 15)
    ax[1].set_xticks(np.arange(0,len(df)))
    ax[0].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticks(np.arange(0,len(df)))
    ax[2].set_xticklabels(df['systematic'].values.tolist(), rotation = 25)
    ax[2].set_xlabel('Systematic Set', size = 13)
    fig.suptitle('Robustness of Resolution', size = 20)    
    ax[0].set_title('Zenith')
    ax[1].set_title('Energy')
    ax[2].set_title('Angular Resolution')

    resolution_filename = generate_file_name(plot_type = 'resolution')
    fig.savefig('plots/' +  resolution_filename)
    
    #### BIAS
    fig, ax = plt.subplots(3,1,figsize=(11.69,8.27))
    
    ax[0].errorbar(np.arange(0,len(df)), df['dynedge_bias_zenith'] - df['dynedge_bias_zenith'][0],marker='o', markeredgecolor='black', label = 'GCN')
    ax[1].errorbar(np.arange(0,len(df)), df['dynedge_bias_energy'] - df['dynedge_bias_energy'][0],marker='o', markeredgecolor='black', label = 'GCN')
    ax[0].errorbar(np.arange(0,len(df)), df['retro_bias_zenith']   - df['retro_bias_zenith'][0],marker='o', markeredgecolor='black', label = 'RetroReco')
    ax[1].errorbar(np.arange(0,len(df)), df['retro_bias_energy']   - df['retro_bias_energy'][0],marker='o', markeredgecolor='black', label = 'RetroReco')
    
    ax[2].errorbar(np.arange(0,len(df)), df['dynedge_bias_angular_res']-df['dynedge_bias_angular_res'][0],marker='o', markeredgecolor='black', label = 'GCN')
    ax[2].errorbar(np.arange(0,len(df)), df['retro_bias_angular_res']  -df['retro_bias_angular_res'][0],marker='o', markeredgecolor='black', label = 'RetroReco')
    


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
    fig.suptitle('Robustness of Bias', size = 20)
    ax[0].set_title('Zenith')
    ax[1].set_title('Energy')
    ax[2].set_title('Angular Resolution')
    
    fig.text(0.04, 0.5, 'Variation w.r.t. Nominal Bias', va='center', rotation='vertical', size = 15)

    bias_filename = generate_file_name(plot_type = 'bias')
    fig.savefig('plots/' + bias_filename)
    
    move_old_plots(bias_filename, resolution_filename)
    return 

if __name__ == '__main__':
    #data_nominal = 'plot_data_nominal_model.csv'
    data_folder = '/groups/hep/pcs557/phd/paper/paper_data/data'

    make_robustness_plots(data_folder, from_csv = True)
