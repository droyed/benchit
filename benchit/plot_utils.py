import matplotlib
import multiprocessing
import platform
import sys
from cpuinfo import get_cpu_info
from psutil import virtual_memory
from types import ModuleType
import importlib
from collections import OrderedDict
import matplotlib.style as style
matplotlib.use('Qt5Agg') # change the matplotlib backend here
style.use('fivethirtyeight')  # choose other styles from style.available
import matplotlib.pyplot as plt


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
        Most relevant system specifications.
    """

    return OrderedDict([('CPU', get_cpu_info()['brand'] + ', ' + str(multiprocessing.cpu_count()) + ' Cores'),
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
    cpu = _bold_latex("CPU :") + d['CPU'] + ', ' + \
        _bold_latex("Mem (GB) :") + d['Memory (GB)'] + ', ' + \
        _bold_latex("ByteO :") + d['ByteOrder']
    kernel_os = _bold_latex("Kernel, OS : ") + d['Kernel-OS']
    python_modules = _bold_latex("Python : ") + d['Python']
    if modules is not None:
        mod = _get_module_versions(modules)
        modules_info = ', '.join([_bold_latex(k + ": ") + v for (k, v) in mod.items()])
        python_modules += ', ' + modules_info
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


def _add_specs_as_title(ax, modules=None):
    """
    Add title with system specifications into an axes plot.

    Parameters
    ----------
    ax : AxesSubplot
        Plot into which the title is to be inserted.
    modules : dict
        Dictionary of modules.

    Returns
    -------
    None
        NA.
    """

    top_offset_buffer = 0.01

    p1, p2, p3 = _latex_formatted_specsinfo(modules=modules)
    L = max([len(i) for i in [p1, p2]])
    p3_split = _splitstr(p3, L)
    p3_split = [s.strip() for s in p3_split]

    # Use the xticklabels fontsize for the title as well
    FSZ = ax.get_xticklabels()[0].get_fontsize()
    title = ax.set_title("\n".join([p1, p2] + p3_split), loc='left', fontsize=FSZ)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.pause(0.1)

    fig = ax.figure
    W1, W2 = title.get_window_extent().size[1], fig.get_window_extent().height
    fig.subplots_adjust(top=1 - W1 / W2 - top_offset_buffer)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.001)
    plt.show(block=False)
    return


def _add_specs_as_textbox(ax, y_offset=0.85, modules=None):
    """
    Add textbox with system specifications into an axes plot.

    Parameters
    ----------
    ax : AxesSubplot
        Plot into which the textbox is to be inserted.
    y_offset : float
        Y position of the textbox.

    Returns
    -------
    None
        NA.
    """

    top_offset_buffer = 0.01

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    fig = ax.figure

    p1, p2, p3 = _latex_formatted_specsinfo(modules=modules)
    L = max([len(i) for i in [p1, p2]])
    p3_split = _splitstr(p3, L)
    specs_txt2 = '\n'.join([p1, p2] + p3_split)

    fig.text(0.3, y_offset, specs_txt2)
    fig.subplots_adjust(top=y_offset - top_offset_buffer)
    plt.pause(0.001)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.pause(0.001)

    plt.show(block=False)
    return


def extract_modules_from_globals(glb, mode='valid'):
    """
    Get modules from globals dict

    Parameters
    ----------
    glb : dict
        Dictionary containing the modules
    mode : str, optional
        Must be one of - `'valid'`, `'all'`

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
