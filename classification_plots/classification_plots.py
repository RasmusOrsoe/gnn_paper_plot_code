import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import sqlite3
from scipy.interpolate import interp1d


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

    fpr_lvl7, tpr_lvl7, _ = roc_curve(data['neutrino'], data['neutrino_pred'])  
    fpr_retro_lvl7, tpr_retro_lvl7, _ = roc_curve(data['neutrino'], data['L7_MuonClassifier_FullSky_ProbNu'])       

    fig = plt.figure(figsize = (10,8))
    plt.title('Signal Classification', size = 30)
    plt.xlabel('False Positive Rate', size = 30)
    plt.ylabel('True Positive Rate', size = 30)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    auc_score_lvl7 = auc(fpr_lvl7,tpr_lvl7)
    auc_score_retro_lvl7 = auc(fpr_retro_lvl7,tpr_retro_lvl7)
    col_labels=['AUC']
    
    y_lvl7, x_lvl7, y_retro_lvl7, x_retro_lvl7  = CalculatePoint(0.7, data)
    plt.plot(np.repeat(x_retro_lvl7, 1000),np.arange(0,1,0.001), '--', color = 'red')
    plt.plot(fpr_retro_lvl7,tpr_retro_lvl7, label = 'Current BDT \n AUC: %s'%round(auc_score_retro_lvl7,5), color = 'orange', lw = 2)
    plt.plot(fpr_lvl7, tpr_lvl7, label = 'GNN \n AUC: %s'%round(auc_score_lvl7,5), color = 'blue', lw = 2)

    #plt.plot(np.repeat(x_retro_lvl4, 1000),np.arange(0,1,0.001), '--', color = 'black')
    plt.scatter(x_retro_lvl7,y_retro_lvl7, color = 'orange')

    plt.plot(x_retro_lvl7, tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))], 'o', color = 'blue')


    plt.plot(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))], y_retro_lvl7, 'o', color = 'blue')

    ### LVL7 ANNOTIATIONS
    plt.annotate(
        '                       (%s,%s)'%(str(round(fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))],4)), str(round(y_retro_lvl7[0],4))),
        color = 'blue',
        size = 13,
        xy=( fpr_lvl7[ np.argmin(abs(tpr_lvl7 - y_retro_lvl7))], y_retro_lvl7[0]), xycoords='data',
        xytext=(85, 53), textcoords='offset points',
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=-0.2",
                        color = 'blue'))

    plt.annotate(
        '                       (%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(y_retro_lvl7[0],4))),
        color = 'orange',
        size = 13,
        xy=( x_retro_lvl7[0], y_retro_lvl7[0]), xycoords='data',
        xytext=(60.2, 40), textcoords='offset points',
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=-0.2",
                        color = 'darkorange'))

    plt.annotate(
        '                       (%s,%s)'%(str(round(x_retro_lvl7[0],4)), str(round(tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))],4))),
        color = 'blue',
        size = 13,
        xy=(x_retro_lvl7[0],tpr_lvl7[ np.argmin(abs(fpr_lvl7 - x_retro_lvl7))]), xycoords='data',
        xytext=(55, -80), textcoords='offset points',
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3,rad=-0.2",
                        color = 'blue'))

    plt.text(x_retro_lvl7[0] - 0.03, 0.87, '0.70 Score Threshold', rotation = 'vertical', color = 'red', fontsize = 15)


    plt.legend(fontsize = 15)
    plt.ylim([0.65,1])
    fig.savefig("roc.png")

def MakeTrackCascadePlot(data):
    
    data = remove_muons(data)
    print(sum(data['track']==1))
    print(sum(data['track']==0))
    fig = plt.figure(figsize = (10,8))
    plt.title('Track/Cascade', size = 20)
    plt.xlabel('False Positive Rate', size = 40)
    plt.ylabel('True Positive Rate', size = 40)

    fpr, tpr, _ = roc_curve(data['track'], data['track_pred'])  
    fpr_retro, tpr_retro, _ = roc_curve(data['track'], data['L7_PIDClassifier_FullSky_ProbTrack'])        

    auc_score = auc(fpr,tpr)
    auc_score_retro = auc(fpr_retro,tpr_retro)

    plt.plot(fpr_retro,tpr_retro, label = 'BDT AUC: %s'%(round(auc_score_retro,3)), color = 'orange')
    plt.plot(fpr, tpr, label = 'dynedge AUC: %s'%(round(auc_score,3)), color = 'blue')
    plt.legend()
    fig.savefig('track_cascade.png')


        




data = pd.read_csv('/groups/hep/pcs557/github/gnn_paper_plot_code/data/0000/everything.csv')
MakeBackgroundSignalPlot(data)
MakeTrackCascadePlot(data)
