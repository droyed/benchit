import os
import sys
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from benchit.utils import extract_modules_from_globals

def mkdir(dirn):
    if not os.path.exists(dirn):
        os.makedirs(dirn)

# Freshly create output dir
def newmkdir(P):
    if os.path.isdir(P):
        shutil.rmtree(P)    
    mkdir(P)

def setup_outdir(bkend, mainoutdir = 'testing_backends_outputs'):
    outdir_pylevel_dirname = 'Py_'+sys.version.replace('.','')[:3]
    outdir_pylevel = os.path.join(mainoutdir, outdir_pylevel_dirname)
    outdir = os.path.join(outdir_pylevel, bkend.replace('/',''))
    
    mkdir(mainoutdir)
    mkdir(outdir_pylevel)
    newmkdir(outdir)
    return outdir

def get_sample_timings_obj_from_json(json_file):
    df = pd.read_json(json_file)
    df.index.name = 'Len'
    df.columns.name = 'Functions'
    return df

def save_all_params_figures(t, outdir):    
    #### Original syntax
    # def plot(self, 
    #          set_xticks_from_index=True,
    #          xlabel=None,
    #          ylabel=None,
    #          colormap='jet',
    #          logx=False,
    #          logy=None,
    #          grid=True,
    #          linewidth = 2,
    #          rot = None,
    #          dpi = None,
    #          fontsize = 14,
    #          specs_fontsize = None,
    #          tick_fontsize = None,
    #          label_fontsize = None,
    #          legend_fontsize = None,
    #          figsize = None,
    #          specs_position = 'left',
    #          debug_plotfs = False,
    #          pause_timefs = 0.1,
    #          modules=None,
    #          save=None,
    #          **kwargs):
    
    # Input args (single)
    set_xticks_from_index=True;
    xlabel=None;
    ylabel=None;
    colormap='jet';
    logx=True;
    logy=None;
    grid=True;
    linewidth = 2;
    rot = None;
    dpi = None;
    fontsize = 14;
    specs_fontsize = None;
    tick_fontsize = None;
    label_fontsize = None;
    legend_fontsize = None;
    figsize = None;
    specs_position = 'left';
    debug_plotfs = False;
    pause_timefs = 0.1;
    modules=None;
    
    
    # Input args combinations
    set_xticks_from_index_vals = [False, True]
    xlabel_vals = [None, 'Custom xlabel']
    ylabel_vals = [None, 'Custom ylabel']
    colormap_vals = [None, 'jet', 'gray']
    logx_vals = [False, True]
    logy_vals = [False, True]
    grid_vals = [False, True]
    linewidth_vals = [2, 4, 10]
    rot_vals = [None, 0, 90]
    dpi_vals = [None, 100, 200]
    fontsize_vals = [2, 10, 14, 20]
    specs_fontsize_vals = [None, 2, 10, 20]
    tick_fontsize_vals = [None, 2, 10, 20]
    label_fontsize_vals = [None, 2, 10, 20]
    legend_fontsize_vals = [None, 2, 10, 20]
    figsize_vals = [None,
                    (5,2), (20,2),
                    (5,5), (20,5),
                    (5,10), (20,10)]
    specs_position_vals = ['left', 'center', 'right']
    debug_plotfs_vals = [False, True]
    pause_timefs_vals = [0.001, 0.01, 0.1, 0.5, 1]
    modules_vals = [None, extract_modules_from_globals(globals())]
    
    
    # Input args combinations as dict
    params = {'set_xticks_from_index':set_xticks_from_index_vals,
              'xlabel':xlabel_vals,
              'ylabel':ylabel_vals,
              'colormap':colormap_vals,
              'logx':logx_vals,
              'logy':logy_vals,
              'grid':grid_vals,
              'linewidth':linewidth_vals,
              'rot':rot_vals,
              'dpi':dpi_vals,
              'fontsize':fontsize_vals,
              'specs_fontsize':specs_fontsize_vals,
              'tick_fontsize':tick_fontsize_vals,
              'label_fontsize':label_fontsize_vals,
              'legend_fontsize':legend_fontsize_vals,
              'figsize':figsize_vals,
              'specs_position':specs_position_vals,
              'debug_plotfs':debug_plotfs_vals,
              'pause_timefs':pause_timefs_vals,
              'modules':modules_vals}
    
    # Input args (single) as dict
    params_singlevals = {'set_xticks_from_index':set_xticks_from_index,
              'xlabel':xlabel,
              'ylabel':ylabel,
              'colormap':colormap,
              'logx':logx,
              'logy':logy,
              'grid':grid,
              'linewidth':linewidth,
              'rot':rot,
              'dpi':dpi,
              'fontsize':fontsize,
              'specs_fontsize':specs_fontsize,
              'tick_fontsize':tick_fontsize,
              'label_fontsize':label_fontsize,
              'legend_fontsize':legend_fontsize,
              'figsize':figsize,
              'specs_position':specs_position,
              'debug_plotfs':debug_plotfs,
              'pause_timefs':pause_timefs,
              'modules':modules}

    # params_singlevals = {'dpi':dpi,
    #           'fontsize':fontsize,
    #           'logx':logx,
    #           'specs_fontsize':specs_fontsize}
    
    # # Testing
    # params_singlevals = {'set_xticks_from_index':set_xticks_from_index,
    #           'logx':True,
    #           'figsize':figsize,
    #           'modules':modules}    
    
    # Save the one with default params
    ax = t.plot(**params_singlevals)
    savepath = os.path.join(outdir, '1_defaults.jpg')
    ax.figure.savefig(savepath, bbox_inches='tight')
    plt.close('all')
         
    for k,vals in params_singlevals.items():
        orgval = params_singlevals[k]
        vals = params[k]
        for iterID,val in enumerate(vals):
            if isinstance(val, list):
                savepath = os.path.join(outdir, k+' : '+str(iterID)+'.jpg')
            else:
                savepath = os.path.join(outdir, k+' : '+str(val)+'.jpg')
            print('===> Combination :')
            params_singlevals[k] = val
            pprint(params_singlevals)
        
            ax = t.plot(**params_singlevals)
            ax.figure.savefig(savepath, bbox_inches='tight')
            plt.close('all')
    
        params_singlevals[k] = orgval

        params_singlevals['dpi'] = 100
        
    return
