import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy.special import expit
import sqlite3
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.use('pdf')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def remove_muons(data):
    data = data.loc[abs(data['pid'] != 13), :]
    data = data.sort_values('event_no').reset_index(drop = True)
    return data

def CalculatePoint(threshold,results):
    tpr_retro = []
    fpr_retro = []
    tpr = []
    fpr = []
    
    key = 'neutrino'

    index_retro   = results['L7_MuonClassifier_FullSky_ProbNu'] >= threshold
    index2  = ~index_retro

    results['L7_MuonClassifier_FullSky_ProbNu'][index_retro] = 1
    results['L7_MuonClassifier_FullSky_ProbNu'][index2] = 0
    
    tp_retro    = sum(results['neutrino'][index_retro] == 1)/(sum(results[key][index_retro] == 1) +sum(results[key][index2] == 1) )
    fp_retro    = sum(results['neutrino'][index_retro] == 0)/(sum(results[key][index_retro] == 0) +sum(results[key][index2] == 0) )
    tpr_retro.append(tp_retro)
    fpr_retro.append(fp_retro)

    index   = results['neutrino_pred'] >= threshold
    index2  = ~index

    results['neutrino_pred'][index] = 1
    results['neutrino_pred'][index2] = 0
    
    tp    = sum(results['neutrino'][index] == 1)/(sum(results[key][index] == 1) +sum(results[key][index2] == 1) )
    fp    = sum(results['neutrino'][index] == 0)/(sum(results[key][index] == 0) +sum(results[key][index2] == 0) )
    tpr.append(tp_retro)
    fpr.append(fp_retro)
    return tpr, fpr, tpr_retro, fpr_retro

def CalculateFPRFraction(fpr, tpr,fpr_retro, tpr_retro):
    common_tpr = np.arange(0,0.8,0.05)
    retro = pd.DataFrame({'fpr':fpr_retro, 'tpr':tpr_retro}).drop_duplicates(subset=['tpr']).sort_values('tpr').reset_index(drop = True)
    data = pd.DataFrame({'fpr':fpr, 'tpr':tpr}).drop_duplicates(subset=['tpr']).sort_values('tpr').reset_index(drop = True)
    print(retro.max())
    print(data.max())
    f = interp1d(x = data['tpr'], y = data['fpr'], kind = 'cubic')
    f_retro = interp1d(x = retro['tpr'], y = retro['fpr'], kind = 'cubic')
    ratio = 1 - (f(common_tpr)/f_retro(common_tpr))
    return ratio

def CalculateTPRFraction(fpr, tpr,fpr_retro, tpr_retro):
    common_fpr = np.arange(0,0.8,0.05)
    retro = pd.DataFrame({'fpr':fpr_retro, 'tpr':tpr_retro}).drop_duplicates(subset=['fpr']).sort_values('tpr').reset_index(drop = True)
    data = pd.DataFrame({'fpr':fpr, 'tpr':tpr}).drop_duplicates(subset=['fpr']).sort_values('tpr').reset_index(drop = True)
    f = interp1d(x = data['fpr'], y = data['tpr'], kind = 'cubic')
    f_retro = interp1d(x = data['fpr'], y = retro['tpr'], kind = 'cubic')
    ratio = f(common_fpr)/f_retro(common_fpr) - 1
    return ratio
    
def ApplyTrackLabel(data,db):
    data = data.sort_values('event_no').reset_index(drop= True)
    with sqlite3.connect(db) as con:
        query = 'select event_no, pid, interaction_type from truth where event_no in %s'%str(tuple(data['event_no']))
        truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop= True)

    
    
    track = (abs(truth['pid']) == 14) & (truth['interaction_type'] == 1)
    data['track'] = track.astype(int).copy()
    return data
    



def MakeBackgroundSignalPlot(data):
    width = 3.176
    height = 2.388
    
    fpr_lvl7, tpr_lvl7, _ = roc_curve(data['neutrino'], data['neutrino_pred'])  
    fpr_retro_lvl7, tpr_retro_lvl7, _ = roc_curve(data['neutrino'], data['L7_MuonClassifier_FullSky_ProbNu'])       

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
    plt.title('Signal Classification', size = 8)
    plt.xlabel('False Positive Rate', size = 8)
    plt.ylabel('True Positive Rate', size = 8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    

    auc_score_lvl7 = auc(fpr_lvl7,tpr_lvl7)
    auc_score_retro_lvl7 = auc(fpr_retro_lvl7,tpr_retro_lvl7)
   
    y_lvl7, x_lvl7, y_retro_lvl7, x_retro_lvl7  = CalculatePoint(0.7, data)
    plt.text(x_retro_lvl7[0] - 0.06, 0.86, '0.70 ' +'$\\nu_{\\alpha}$' + ' Score', rotation = 'vertical', color = 'red', fontsize = 7)
    plt.plot(np.repeat(x_retro_lvl7, 1000),np.arange(0,1,0.001), '--', color = 'red')
    plt.plot(fpr_retro_lvl7,tpr_retro_lvl7, label = 'Current BDT \n AUC: %s'%round(auc_score_retro_lvl7,5), color = 'orange', lw = 2)
    plt.plot(fpr_lvl7, tpr_lvl7, label = 'GNN \n AUC: %s'%round(auc_score_lvl7,5), color = 'blue', lw = 2)


    plt.plot(x_retro_lvl7, tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))], '^', color = 'lightblue')
    plt.plot(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))], y_retro_lvl7, 'o', color = 'lightblue')
    plt.plot(x_retro_lvl7,y_retro_lvl7,'o',color = 'darkorange')
    ### LVL7 ANNOTIATIONS
    plt.plot(0.24, 0.666, 'o', color = 'lightblue')
    plt.plot(0.24, 0.696, 'o', color = 'darkorange')
    plt.plot(0.24, 0.726, '^', color = 'lightblue')
    plt.text(0.27, 0.66, '(%s,%s)'%(str(round(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))],4)), str(round(y_retro_lvl7[0],4))), color = 'blue', fontsize = 8)
    plt.text(0.27, 0.69, '(%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(y_retro_lvl7[0],4))), color = 'orange', fontsize = 8)
    plt.text(0.27, 0.72,'(%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))],4))), color = 'blue', fontsize = 8)


    
    plt.legend(fontsize = 6)
    plt.ylim([0.65,1])
    fig.savefig('roc.pdf',bbox_inches="tight")

def calculate_track_cascade_density(data, is_retro):
    if is_retro:
        key = 'L7_PIDClassifier_FullSky_ProbTrack'
    else:
        key = 'track_pred'
    thresholds = np.arange(0,1.01,0.01)
    total_tracks = sum(data['track'] == 1)
    total_cascades = sum(data['track'] == 0)

    n_tracks = []
    n_cascades = []
    for threshold in thresholds:
        n_tracks.append(sum(data['track'][data[key]>= threshold] == 1)/total_tracks)
        n_cascades.append(sum(data['track'][data[key]>= threshold] == 0)/total_cascades)
    return n_tracks, n_cascades, thresholds


def MakeTrackCascadePlot(data, mode = 'physical'):
    data = remove_muons(data)
    data['track_pred'] = expit(data['track_pred'])
    width = 3.176
    height = 2.388
    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    if mode == 'roc':
        plt.title('Track/Cascade', size = 8)
        plt.xlabel('False Positive Rate', size = 8)
        plt.ylabel('True Positive Rate', size = 8)

        fpr, tpr, threshold = roc_curve(data['track'], data['track_pred'])  
        fpr_retro, tpr_retro, threshold_retro = roc_curve(data['track'], data['L7_PIDClassifier_FullSky_ProbTrack'])        

        auc_score = auc(fpr,tpr)
        auc_score_retro = auc(fpr_retro,tpr_retro)

        plt.plot(fpr_retro,tpr_retro, label = 'Current BDT AUC: %s'%(round(auc_score_retro,3)), color = 'orange')
        plt.plot(fpr, tpr, label = 'GNN AUC: %s'%(round(auc_score,3)), color = 'blue')
        plt.legend(fontsize = 6)
        fig.savefig('track_cascade.pdf',bbox_inches="tight")
    if mode == 'auc_vs_E':
        plt.title('Track/Cascade Classification', size = 8)
        plt.xlabel('False Positive Rate', size = 8)
        plt.ylabel('True Positive Rate', size = 8)
        bins = np.arange(0,3.10, 0.10)
        auc_score = []
        auc_score_retro = []
        mean_energy_in_bin = []
        for i in range(1,len(bins)):
            data_sliced = data.loc[(np.log10(data['energy'])> bins[i-1]) & (np.log10(data['energy'])< bins[i]),:].reset_index(drop = True)
            fpr, tpr, _ = roc_curve(data_sliced['track'], data_sliced['track_pred'])  
            fpr_retro, tpr_retro, _ = roc_curve(data_sliced['track'], data_sliced['L7_PIDClassifier_FullSky_ProbTrack'])        
            
            auc_score.append(auc(fpr,tpr))
            auc_score_retro.append(auc(fpr_retro,tpr_retro))
            mean_energy_in_bin.append(np.mean(np.log10(data_sliced['energy'])))

        print(mean_energy_in_bin)
        plt.scatter(mean_energy_in_bin,auc_score_retro, label = 'Current BDT', color = 'orange')
        plt.scatter(mean_energy_in_bin, auc_score, label = 'GNN', color = 'blue')
        plt.legend(fontsize = 6)
        fig.savefig('track_cascade_energy_auc.pdf',bbox_inches="tight")
    if mode == 'physical':
        plt.title('Track/Cascade Classification', size = 8)
        plt.xlabel('False Positive Rate', size = 8)
        plt.ylabel('True Positive Rate', size = 8)

        n_tracks, n_cascades, thresholds = calculate_track_cascade_density(data, is_retro = False)
        n_tracks_retro, n_cascades_retro, thresholds = calculate_track_cascade_density(data, is_retro = True)
 
        plt.plot(thresholds,n_tracks_retro, label = 'Current BDT Track', color = 'orange')
        plt.plot(thresholds,n_cascades_retro, label = 'Current BDT Cascades', ls = '--', color = 'orange')
        plt.plot(thresholds, n_tracks, label = 'GNN Track', color = 'blue')
        plt.plot(thresholds, n_cascades, label = 'GNN Cascades', ls = '--', color = 'blue')
        plt.legend(fontsize = 6)
        
        fig.savefig('track_cascade_physical.pdf',bbox_inches="tight")


def MakeCombinedPlot(signal_data, track_data):
    width = 3.176
    height = 2*2.388
    
    fpr_lvl7, tpr_lvl7, _ = roc_curve(signal_data['neutrino'], signal_data['neutrino_pred'])  
    fpr_retro_lvl7, tpr_retro_lvl7, _ = roc_curve(signal_data['neutrino'], signal_data['L7_MuonClassifier_FullSky_ProbNu'])       

    fig, ax = plt.subplots(2,1,constrained_layout = True)
    fig.set_size_inches(width, height)
    ax[0].set_ylabel('True Positive Rate', size = 14)
    #plt.xticks(fontsize=6)
    ax[0].tick_params(axis='x', labelsize=6)

    

    auc_score_lvl7 = auc(fpr_lvl7,tpr_lvl7)
    auc_score_retro_lvl7 = auc(fpr_retro_lvl7,tpr_retro_lvl7)
   
    y_lvl7, x_lvl7, y_retro_lvl7, x_retro_lvl7  = CalculatePoint(0.7, signal_data)
    ax[0].text(x_retro_lvl7[0] - 0.06, 0.86, '0.70 ' +'$\\nu_{\\alpha}$' + ' Score', rotation = 'vertical', color = 'red', fontsize = 7)
    ax[0].plot(np.repeat(x_retro_lvl7, 1000),np.arange(0,1,0.001), '--', color = 'red')
    ax[0].plot(fpr_retro_lvl7,tpr_retro_lvl7, label = 'Current BDT \n AUC: %s'%round(auc_score_retro_lvl7,5), color = 'orange', lw = 2)
    ax[0].plot(fpr_lvl7, tpr_lvl7, label = 'GNN \n AUC: %s'%round(auc_score_lvl7,5), color = 'blue', lw = 2)


    ax[0].plot(x_retro_lvl7, tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))], '^', color = 'lightblue')
    ax[0].plot(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))], y_retro_lvl7, 'o', color = 'lightblue')
    ax[0].plot(x_retro_lvl7,y_retro_lvl7,'o',color = 'darkorange')
    ### LVL7 ANNOTIATIONS
    ax[0].plot(0.24, 0.666, 'o', color = 'lightblue')
    ax[0].plot(0.24, 0.696, 'o', color = 'darkorange')
    ax[0].plot(0.24, 0.726, '^', color = 'lightblue')
    ax[0].text(0.27, 0.66, '(%s,%s)'%(str(round(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))],4)), str(round(y_retro_lvl7[0],4))), color = 'blue', fontsize = 8)
    ax[0].text(0.27, 0.69, '(%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(y_retro_lvl7[0],4))), color = 'orange', fontsize = 8)
    ax[0].text(0.27, 0.72,'(%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))],4))), color = 'blue', fontsize = 8)


    
    ax[0].legend(fontsize = 6)
    ax[0].set_ylim([0.65,1])
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax[0].set_xticklabels(empty_string_labels)


    ## Tracks
    data = remove_muons(track_data)
    data['track_pred'] = expit(data['track_pred'])

    #plt.title('Track/Cascade', size = 8)
    ax[1].set_xlabel('False Positive Rate', size = 14)
    ax[1].set_ylabel('True Positive Rate', size = 14)

    fpr, tpr, threshold = roc_curve(data['track'], data['track_pred'])  
    fpr_retro, tpr_retro, threshold_retro = roc_curve(data['track'], data['L7_PIDClassifier_FullSky_ProbTrack'])        

    auc_score = auc(fpr,tpr)
    auc_score_retro = auc(fpr_retro,tpr_retro)

    ax[1].plot(fpr_retro,tpr_retro, label = 'Current BDT AUC: %s'%(round(auc_score_retro,3)), color = 'orange')
    ax[1].plot(fpr, tpr, label = 'GNN AUC: %s'%(round(auc_score,3)), color = 'blue')

    ax[1].legend(fontsize = 6)


    ax[0].text(0.63,0.85, 'Signal', fontsize = 12)
    ax[1].text(0.5,0.40, 'Track/Cascade', fontsize = 12)
    fig.savefig('combined.pdf',bbox_inches="tight")
    return


        




signal_data = pd.read_csv('/groups/hep/pcs557/phd/paper/paper_data/data/0000/signal.csv')
#MakeBackgroundSignalPlot(data)
track_data = pd.read_csv('/groups/hep/pcs557/phd/paper/paper_data/data/0000/track_cascade.csv')
#MakeTrackCascadePlot(data, mode = 'roc')
MakeCombinedPlot(signal_data = signal_data, track_data= track_data)