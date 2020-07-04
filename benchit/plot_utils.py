import matplotlib
import multiprocessing
import platform
import sys
from cpuinfo import get_cpu_info
from psutil import virtual_memory
from types import ModuleType
import importlib
from collections import OrderedDict
import numpy as np
import matplotlib.colors as colors
import colorsys
import matplotlib.style as style
style.use('fivethirtyeight')  # choose other styles from style.available
import matplotlib.pyplot as plt


def _add_specs_as_title(dfp, specs_fontsize=None, _FULLSCREENDEBUG=False, modules=None):
    figManager = plt.get_current_fig_manager()
    done, status = _full_screen_or_toggle(figManager)
    if _FULLSCREENDEBUG:
        print('Fullscreen debug status : '+status)
        print('Backend : '+matplotlib.get_backend())    
        print('Fullscreen done : '+str(done))
    
    if specs_fontsize is None:
        specs_fontsize = dfp.get_xticklabels()[0].get_fontsize()
    
    plt.pause(0.001)

    # Get plot specifications
    p1, p2, p3 = _latex_formatted_specsinfo(modules=modules)
    L = max([len(i) for i in [p1, p2]])
    p3_split = _splitstr(p3, L)
    p3_split = [s.strip() for s in p3_split]    
    plt_specsinfo = "\n".join([p1, p2] + p3_split)
    
    # Set specs as title and show
    dfp.set_title(plt_specsinfo, loc='left', fontsize=specs_fontsize)
    plt.pause(0.001)
    plt.show(block=False)    
    return dfp


def _full_screen_or_toggle(figManager):
    done = True
    status = ''
    try:
        figManager.window.showMaximized()
        status += '\nStatus : figManager.window.showMaximized worked.'
    except Exception as errMsg:
        status += '\nfigManager.window.showMaximized failed. Reason : '+str(errMsg)
        try:
            figManager.resize(*figManager.window.maxsize())
            status += '\nfigManager.resize worked.'
        except Exception as errMsg:
            status += '\nfigManager.resize failed. Reason : '+str(errMsg)
            try:
                figManager.full_screen_toggle()  
                status += '\nfigManager.full_screen_toggle worked.' 
            except Exception as errMsg:
                status += '\nfigManager.full_screen_toggle failed; no fullscreen applied. Reason : '+str(errMsg)
                done = False
    
    return done, status

def _get_specsinfo():
    """
    Gets system specifications.

    Parameters
    ----------
    None
        NA.

    Returns
    -------
    dict
        Dictionary listing most relevant system specifications.
    """

    cpuinfo_ = get_cpu_info()
    if 'brand' in cpuinfo_:
        CPU_brand = cpuinfo_['brand']
    elif 'brand_raw' in cpuinfo_:
        CPU_brand = cpuinfo_['brand_raw']
    else:
        CPU_brand = 'CPU - NA'
    return OrderedDict([('CPU', CPU_brand + ', ' + str(multiprocessing.cpu_count()) + ' Cores'),
                        ('Memory (GB)', str(round(virtual_memory().total / 1024.**3, 1))),
                        ('ByteOrder', sys.byteorder.capitalize()),
                        ('Kernel-OS', platform.platform()),
                        ('Python', platform.python_version())])


def _get_module_version(mod):
    """
    Gets module version from module.

    Parameters
    ----------
    mod : module
        Input module whose version ID is to be extracted.

    Returns
    -------
    str
        Extracted module version.
    """

    parent_module = importlib.import_module(mod.__name__.split('.')[0])
    if "__version__" in dir(parent_module):
        return parent_module.__version__
    else:
        return "NA"


def _get_module_versions(mods):
    """
    Gets module versions from dict of modules.

    Parameters
    ----------
    mods : dict
        Dictionary containing the modules whose version IDs are to be extracted.

    Returns
    -------
    dict
        Extracted module versions.
    """

    out = {}
    for i in mods:
        name = i.__name__.split('.')[0].capitalize()
        version = _get_module_version(i)
        out.update({name: version})
    return out


def print_specs(modules=None):
    """
    Print system specifications.

    Parameters
    ----------
    modules : dict, optional
        Dictionary containing the modules. These are optionally included to
        setup python modules info and printing it.

    Returns
    -------
    None
        NA.

    """

    d = _get_specsinfo()
    for (i, v) in d.items():
        print(i + ' : ' + v)
    if modules is not None:
        mod = _get_module_versions(modules)
        print("Python Module(s) : ")
        for k in sorted(mod.keys()):
            print("    " + k + " : " + mod[k])


def _latex_formatted_specsinfo(modules=None):
    """
    Get latex formatted specifications.

    Parameters
    ----------
    modules : dict
        Dictionary containing the modules. These are optionally included to
        setup python modules info.

    Returns
    -------
    tuple
        CPU, kernel-OS, python modules information in latex format.
    """

    def _bold_latex(s):
        """Get latex bold formmatted version of input string."""
        return r"$\bf{" + s + "}$"

    d = _get_specsinfo()
    cpu = _bold_latex("CPU :") + d['CPU'] + '  ' + _bold_latex("Mem (GB) :") +\
        d['Memory (GB)'] + '  ' + _bold_latex("ByteO :") + d['ByteOrder']
    kernel_os = _bold_latex("Kernel, OS : ") + d['Kernel-OS']
    python_modules = _bold_latex("Python : ") + d['Python']
    if modules is not None:
        mod = _get_module_versions(modules)
        modules_info = '  '.join([_bold_latex(k + ": ") + v for (k, v) in mod.items()])
        python_modules += '  ' + modules_info
    return cpu, kernel_os, python_modules


def _splitstr(m_str, maxlen, delimiter=','):
    """
    Split a string into blocks of strings such that each block is limited to
    a length of maxlen.

    Parameters
    ----------
    m_str : str
        Input string to be split.
    maxlen : int
        Maximum length of each split string.

    Returns
    -------
    List
        List of split strings.
    """

    m_str_split = m_str.split(delimiter)
    lens = [len(i) for i in m_str_split]

    s = 0
    idx = [0]
    for i, l in enumerate(lens):
        s += l
        if s > maxlen:
            idx.append(i)
            s = l
    idx.append(len(lens))
    m_str_split_grp = [m_str_split[i:j] for (i, j) in zip(idx[:-1], idx[1:])]
    return [delimiter.join(i) for i in m_str_split_grp]


def extract_modules_from_globals(glb, mode='valid'):
    """
    Get modules from globals dict.

    Parameters
    ----------
    glb : dict
        Dictionary containing the modules.
    mode : str, optional
        Must be one of - `'valid'`, `'all'`.

    Returns
    -------
    list
        Extracted modules in a list
    """

    kv = zip(glb.keys(), glb.values())
    b = ['__builtin__', '__builtins__']

    modules = [v for (k, v) in kv if k not in b and isinstance(v, ModuleType)]
    unq_modules = list(set(modules))
    if mode == 'valid':
        modules = [l for l in unq_modules if not l.__name__.startswith('_')]
        return [m for m in modules if _get_module_version(m) != 'NA']
    elif mode == 'all':
        return [l for l in unq_modules if not l.__name__.startswith('_')]
    else:
        return Exception('Wrong argument for mode!')


def _truncate_cmap(cmap, Y_thresh=0.65, start_offN = 100):
    """
    Truncate colormap so that we avoid a certain range of Y values in YIQ color space.

    Parameters
    ----------
    cmap : str
        Colormap string.
    Y_thresh : int, optional
        Y threshold value.
    start_offN : int, optional
        Starting number of levels.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap.
    """

    cmap_func = plt.get_cmap(cmap)
    allcolors = cmap_func(np.linspace(0., 1., start_offN))
    mask = np.array([colorsys.rgb_to_yiq(*c[:-1])[0]<=Y_thresh for c in allcolors])
    if ~mask.any():
        return cmap # not truncated
    else:
        return colors.LinearSegmentedColormap.from_list('trunc_cmap', allcolors[mask])
