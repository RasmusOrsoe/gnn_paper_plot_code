import pandas as pd
import os
import sqlite3
from multiprocessing import Pool

def find_systematic_folder(root):
    folders = os.listdir(root)
    systematics = []
    for folder in folders:
        if 'robustness_muon_neutrino' in folder:
            systematics.append(folder)
    return systematics

def add_truth(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select * from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    columns = list(data.columns)
    columns.extend(list(truth.drop(columns = ['event_no']).columns))
    data = pd.concat([data, truth.drop(columns = ['event_no'])], axis = 1, ignore_index= True)
    data.columns = columns
    return data

def add_retro(data, database):
    data = data.sort_values('event_no').reset_index(drop = True)
    with sqlite3.connect(database) as con:
        query = 'select * from RetroReco where event_no in %s'%str(tuple(data['event_no']))
        retro = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)
    columns = list(data.columns)
    columns.extend(list(retro.drop(columns = ['event_no']).columns))
    data = pd.concat([data, retro.drop(columns = ['event_no'])], axis = 1, ignore_index= True)
    data.columns = columns
    return data

def make_csv(settings):
    database, result_folder, outdir = settings
    folders = os.listdir(result_folder)
    available_results = {}
    if '0000' in result_folder:
        for folder in folders:
            if 'dynedge_paper_test_set' in folder:
                target = folder.split('_')[-1]
                test_set =  pd.read_csv(result_folder + '/' + folder + '/results.csv').sort_values('event_no').reset_index(drop = True)
                validation_set = pd.read_csv(result_folder + '/' + 'dynedge_paper_valid_set_%s'%target + '/results.csv').sort_values('event_no').reset_index(drop = True)
                append_this = pd.concat([test_set, validation_set], axis = 0, ignore_index= True).sort_values('event_no').reset_index(drop = True)

                available_results[folder] = append_this
    else:
        for folder in folders:
            if 'dynedge_paper_' in folder:
                    available_results[folder] = pd.read_csv(result_folder + '/' + folder + '/results.csv').sort_values('event_no').reset_index(drop = True)
        
    is_first = True
    for result in available_results.keys():
        if is_first:
            data = available_results[result].drop(columns = ['Unnamed: 0'])
            is_first = False
            columns = list(data.columns)
        else:
            new_data = available_results[result].drop(columns = ['event_no', 'Unnamed: 0'])
            columns.extend(list(new_data.columns))
            data = pd.concat([data, new_data], axis = 1, ignore_index= True)
    data.columns = columns
    data = data.drop(columns = ['energy', 'zenith', 'azimuth', 'XYZ'])
    data = add_retro(data, database)
    data = add_truth(data, database)
    data.to_csv(outdir + '/' + 'everything.csv')
    del data
    return
if __name__ == '__main__':
    root = '/groups/hep/pcs557/phd/paper/regression_results'
    systematics = find_systematic_folder(root)
    count = 1
    settings = []
    for systematic in systematics:
        print ('%s / %s '%(count, len(systematics)))
        database = '/groups/hep/pcs557/GNNReco/data/databases' + '/' + systematic + '/data/' + systematic + '.db'
        folder = root + '/' + systematic 
        outdir = '/groups/hep/pcs557/github/gnn_paper_plot_code' + '/data/' + systematic[-4:]
        os.makedirs(outdir, exist_ok= True)
        settings.append([database, folder, outdir])
        #make_csv(folder, database, outdir)
        count +=1
    #make_csv(settings[0])
    p = Pool(processes = len(settings[0:10]))
    async_result = p.map_async(make_csv, settings[0:10])
    p.close()
    p.join()
    
    p = Pool(processes = len(settings[10:]))
    async_result = p.map_async(make_csv, settings[10:])
    p.close()
    p.join()
    print("Complete")

