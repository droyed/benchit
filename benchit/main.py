import numpy as np
import pandas as pd
import timeit
import functools
import warnings
from tqdm import trange
import colorsys
import matplotlib
import matplotlib.colors as colors
import matplotlib.style as style
import matplotlib.pyplot as plt
from benchit.utils import specs, specs_short

style.use('fivethirtyeight')
warnings.simplefilter('always', UserWarning)


# Parameters
_TIMEOUT = 0.2
_NUM_REPEATS = 5
_ENVIRON = 'normal'


def setparams(timeout=0.2, rep=5, environ='normal'):
    """
    Set parameters for benchit.

    Parameters
    ----------
    timeout : float or int, optional
        Sets up timeout while looping with timeit that decides when to exit benchmarking for current iteration setup.
    rep : float or int, optional
        Sets up number of repetitions as needed to select the best timings among them as final runtime number for current iteration setup.
    environ : str, optional
        String that sets up environment given the current setup with global variable _ENVIRON.

    Returns
    -------
    None
        NA.

    """

    global _TIMEOUT
    global _NUM_REPEATS
    global _ENVIRON

    if environ not in ['notebook', 'normal']:
        raise ValueError("Invalid environ value. Must be 'notebook' or 'normal'.")

    if environ == 'notebook':
        print('Notebook environment set! Use "fontsize" & "figsize" args with plot method for better viewing experience.')

    _TIMEOUT = timeout
    _NUM_REPEATS = rep
    _ENVIRON = environ
    return


def fullscreenfig(ax, pause_timefs, print_info=False):
    """
    Make the current figure fullscreen.

    Parameters
    ----------
    figManager : matplotlib backend FigureManager
        Figure manager of the current figure.

    Returns
    -------
    done : bool
        Boolean flag that is True or False if full-screen worked or not respectively.
        Note that for inlined plots on notebooks, this won't work.
    status : str
        Status message for debugging.
    """

    done = True
    status = ''

    figManager = plt.get_current_fig_manager()
    backend = matplotlib.get_backend()

    if backend in matplotlib.rcsetup.non_interactive_bk:
        status = 'Backend is non-interactive. So no trials were made to fullscreen and hence no pause.'
        info = {'done': False, 'status': status, 'backend': backend}
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                figManager.window.showMaximized()
                status += '\nStatus : figManager.window.showMaximized worked.'
            except Exception as errMsg:
                status += '\nfigManager.window.showMaximized failed. Reason : ' + str(errMsg)
                try:
                    figManager.resize(*figManager.window.maxsize())
                    status += '\nfigManager.resize worked.'
                except Exception as errMsg:
                    status += '\nfigManager.resize failed. Reason : ' + str(errMsg)
                    try:
                        figManager.full_screen_toggle()
                        status += '\nfigManager.full_screen_toggle worked.'
                    except Exception as errMsg:
                        status += '\nfigManager.full_screen_toggle failed; no fullscreen applied. Reason : ' + str(errMsg)
                        done = False

            # Pause before the screen becomes fullscreen
            try:
                plt.pause(interval=pause_timefs)
            except Exception as errMsg:
                status += '\nPause failed. Reason : ' + str(errMsg)

            info = {'done': done, 'status': status, 'backend': backend}
            if print_info:
                print('=> Fullscreen debug status : ' + info['status'])
                print('=> Backend : ' + info['backend'])
                print('=> Fullscreen done : ' + str(info['done']))

    return info


def _truncate_cmap(cmap, Y_thresh=0.65, start_offN=100):
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
    mask = np.array([colorsys.rgb_to_yiq(*c[:-1])[0] <= Y_thresh for c in allcolors])
    if ~mask.any():
        return cmap  # not truncated
    else:
        return colors.LinearSegmentedColormap.from_list('trunc_cmap', allcolors[mask])


def _get_timings_perinput(funcs, input_=None):
    """
    Evaluate function calls on a given input to compute the timings.

    Parameters
    ----------
    funcs : list
        List of functions to be timed.
    input_ : list or tuple or None, optional
        It represents one dataset, so it could be fed to a function for evaluation.

    Returns
    -------
    list
        List of timings from benchmarking.

    Notes
    -----
    The timing mechanism is inspired by timeit tools :
        https://docs.python.org/3/library/timeit.html#timeit.Timer.autorange
        https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat

    """

    global _TIMEOUT
    global _NUM_REPEATS

    timings_l = []

    from IPython import get_ipython
    if get_ipython() is None:
        iter_funcs = trange(len(funcs), desc='Loop functions', leave=False)
    else:
        iter_funcs = range(len(funcs))

    for j in iter_funcs:
        f = funcs[j]
        ii = 1
        process_next = True
        while process_next:
            for jj in 1, 2, 5:
                iter_rep = ii * jj
                if input_ is None:
                    t = min(timeit.repeat(functools.partial(f), repeat=_NUM_REPEATS, number=iter_rep))
                else:
                    t = min(timeit.repeat(functools.partial(f, *input_), repeat=_NUM_REPEATS, number=iter_rep))
                if t > _TIMEOUT:
                    process_next = False
                    break
            ii *= 10
        timings_l.append(t / iter_rep)
    return timings_l


def _get_timings(funcs, inputs=None, multivar=False):
    """
    Evaluate function calls on given inputs to compute the timings.

    Parameters
    ----------
    funcs : list
        List of functions to be timed.
    inputs : list or tuple or None, optional
        Each elements of it represents one dataset each.
    multivar : bool, optional
        Decides whether to consider single or multiple variable input for
        feeding into the functions. As such it expects all functions to accept
        inputs in the same format.
        With the value as False, it assumes that every function accepts only
        one input. Hence, each element in inputs is considered as the only
        input to every function call.
        With the value as True, it assumes that every function accepts more
        than one input. Hence, each element in inputs is unpacked and fed to all functions.

    Returns
    -------
    numpy.ndarray
        Lists of timings such that each row is for one dataset and each column
        represents a function call.
    """

    if inputs is None:
        timings = _get_timings_perinput(funcs)
    else:
        l1, l2 = len(inputs), len(funcs)
        timings = np.empty((l1, l2))
        for i in trange(l1, desc='Loop datasets ', leave=False):
            input_ = inputs[i]
            if not multivar:
                input_ = [input_]
            timings[i, :] = _get_timings_perinput(funcs, input_)
    return timings


def _get_possible_indexbys(inputs):
    """
    Get the possible indexby arguments given inputs in a list or dict.

    Parameters
    ----------
    inputs : list or dict
        List or dictionary that holds each dataset as an element.

    Returns
    -------
    list
        List of strings that lists the various index-by options given the type
        and format of input datasets.

    """

    if isinstance(inputs, dict):
        inputs = list(inputs.values())

    # Get input type
    if all([isinstance(i, (np.ndarray, pd.DataFrame)) for i in inputs]):
        in_type = 'array_df'
    elif all([isinstance(i, (list, tuple)) for i in inputs]):
        in_type = 'list_tuple'
    elif all([np.isscalar(i) for i in inputs]):
        in_type = 'scalar'
    else:
        in_type = 'item'

    # Get possible indexby options for given type of inputs
    allowed_indexbys = {'array_df': ['len', 'shape', 'size', 'item'],
                        'list_tuple': ['len', 'item'],
                        'scalar': ['scalar', 'item'],
                        'item': ['item']}

    possible_indexbys = allowed_indexbys[in_type]
    return possible_indexbys


def _get_params(in_, indexby):
    """
    Given inputs in a list or dictionary and a parameter, sets xticklabels, xlabel.

    Parameters
    ----------
    in_ : list or dict
        Each elements of it represents one dataset each.
    indexby : str
        It must be one among - 'auto', 'len', 'shape', 'scalar', 'item'.
        Depending on the fed indexby value and the type of inputs, appropriate
        indexing options are set for timings output. There are customized
        options for - arrays, dataframes, list, tuple, scalar.

    Returns
    -------
    params : dict
        Dictionary that holds plotting paramters for each of the datasets.
        This would be used for setting xticklabels and x-label later on for plotting.
    list
        List of input datasets
    """

    def _get_params_from_list(inputs, indexby):
        """
        Set xticklabels and indexby string based on type and format of inputs,
        given inputs in a list and indexby option.

        inputs : list
            Each elements of it represents one dataset each.
        indexby : str
            See function definition of _get_params for more details.

        Returns
        -------
        list
            List of strings to be used for plotting as xticklabels.
        str
            String that sets indexby parameter to be used in other functions.

        """

        def f_shp(i):
            """Format array shape info."""
            return i.replace('(', '').replace(')', '').replace(', ', 'x').replace(',', '')

        # Get input type and possible indexbys
        possible_indexbys = _get_possible_indexbys(inputs)

        # Get auto(default) indexby value for given type of inputs
        if indexby == 'auto':
            indexby = possible_indexbys[0]

        # Check if its a valid indexby value by looking for match in possible
        # ones
        if indexby not in possible_indexbys:
            posindxby = ["'" + i + "'" for i in possible_indexbys]
            raise ValueError("Invalid indexby value. Possible indexby value(s) : " + ' or '.join(posindxby) + ".")

        # Setup xticklabels
        R = range(len(inputs))
        if indexby == 'len':
            len_index = list(map(len, inputs))
            xticklabels = len_index
        elif indexby == 'shape':
            shp_index = [f_shp(str(i.shape)) for i in inputs]
            xticklabels = shp_index
        elif indexby == 'size':
            shp_index = [i.size for i in inputs]
            xticklabels = shp_index
        elif indexby == 'item':
            xticklabels = R
        elif indexby == 'scalar':
            d = np.array(inputs).dtype
            if np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer):
                xticklabels = inputs
            else:
                xticklabels = inputs
        else:
            pass

        indexby_str = indexby.capitalize()
        return xticklabels, indexby_str

    def _get_params_from_dict(in_):
        """
        Set xticklabels and indexby string based on type and format of inputs,
        given inputs in a dict.

        in_ : dict
            Each elements of it represents one dataset each.

        Returns
        -------
        Same as with _get_params_from_list.

        """

        idx = np.array(list(in_.keys()))
        d = idx.dtype
        if np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer):
            xticklabels, indexby_str = idx, 'Scalar'
        else:
            xticklabels, indexby_str = [str(i) for i in idx], 'item'
        return xticklabels, indexby_str

    if in_ is None:
        inputs = in_
        xticklabels, xlabel = ['NoArg'], 'Case'
    elif isinstance(in_, dict):
        inputs = list(in_.values())
        if indexby == 'auto':
            xticklabels, xlabel = _get_params_from_dict(in_)
        else:
            xticklabels, xlabel = _get_params_from_list(inputs, indexby)
    else:
        inputs = in_
        xticklabels, xlabel = _get_params_from_list(in_, indexby)
    params = {'xticklabels': xticklabels, 'xlabel': xlabel}
    return params, inputs


def timings(funcs, inputs=None, multivar=False, input_name=None, indexby='auto'):
    """
    Evaluate function calls on given input(s) to compute the timing.
    Puts out a dataframe-like object with the input properties being put into
    the header and index names and values.

    Parameters
    ----------
    funcs : list or tuple
        Contains the functions to be timed.
    inputs : list or tuple or None, optional
        Each elements of it represents one dataset each.
    multivar : bool, optional
        Decides whether to consider single or multiple variable input for
        feeding into the functions. As such it expects all functions to accept
        inputs in the same format.
        With the value as False, it assumes that every function accepts only
        one input. Hence, each element in inputs is considered as the only
        input to every function call.
        With the value as True, it assumes that every function accepts more
        than one input. Hence, each element in inputs is unpacked and fed to all functions.
    input_name : str, optional
        String that sets the index name for the output timings dataframe.
        This is used later on with plots to automatically assign x-label.
    indexby : str, optional
        String that sets the index properties for the output timings dataframe.
        Argument value must be one of - `'len'`, `'shape'`, `'item'`, `'scalar'`.

    Returns
    -------
    BenchmarkObj
        Timings stored in a dataframe-like object with each row for each dataset
        and each column represents a function call.

    """

    # Default params
    multiindex = False

    # Setup label parameters
    if multivar and not isinstance(inputs, dict):
        indexby = 'item'
    p, inputs_p = _get_params(inputs, indexby=indexby)
    xticklabels, xlabel_from_inputs = p['xticklabels'], p['xlabel']

    # Get timings dataframe
    if isinstance(funcs, dict):
        t_ = _get_timings(list(funcs.values()), inputs_p, multivar=multivar)
        df_timings = pd.DataFrame(t_, columns=funcs.keys())
    else:
        t_ = _get_timings(funcs, inputs_p, multivar=multivar)
        df_timings = pd.DataFrame(np.atleast_2d(t_), columns=[i.__name__ for i in funcs])

    df_timings.columns.name = 'Functions'

    # Setup index properties in the dataframe
    if isinstance(inputs, dict) and multivar:
        df_timings.index = inputs.keys()
    else:
        df_timings.index = xticklabels

    # Multivar additions for v005 to setup df_timings.index.name
    if isinstance(input_name, list) or isinstance(input_name, tuple):
        input_name_type = 'list_or_tuple'
    elif isinstance(input_name, str):
        input_name_type = 'str'
    else:
        input_name_type = None

    df_nlevels = df_timings.index.nlevels
    if input_name_type == 'str':
        if df_nlevels > 1:
            warnings.warn('It is multivar, but multiindex is not set with input_name '
                          'or when specifying inputs. Hence, creating a regular timings object.', stacklevel=2)

        # Convert MultiIndex dataframe to a regular one by packing in each tuple
        if all([isinstance(dfi, tuple) for dfi in df_timings.index]):
            df_timings.index = list(map(tuple, df_timings.index))
        df_timings.index.name = input_name

    else:
        if multivar and df_nlevels > 1:
            if input_name_type is None:
                df_timings.index.names = ['Arg' + str(i) for i in range(df_nlevels)]
                multiindex = True
            elif input_name_type == 'list_or_tuple':
                if len(input_name) < df_nlevels:
                    raise Exception('For multivar case, input_name must be a list or tuple of input argument names for each argument.')
                df_timings.index.names = input_name[:df_nlevels]
                if df_nlevels > 1:
                    multiindex = True
            else:
                raise Exception('Invalid input_name values for given input argument(s).')
        else:
            if input_name_type is None:
                df_timings.index.name = xlabel_from_inputs
            elif input_name_type == 'list_or_tuple':
                df_timings.index.name = input_name[0]
            else:
                raise Exception('Invalid input_name values for given input argument(s).')

    # High-level warning msg
    if multivar and not multiindex:
        warnings.warn('It is multivar, but multiindex is not set with input_name or when specifying inputs.', stacklevel=2)

    benchObj = BenchmarkObj(df_timings, multivar=multivar, multiindex=multiindex)
    return benchObj


def _assign_mplibrcparams(fontsize, specs_fontsize, tick_fontsize, label_fontsize, legend_fontsize, specs_position, dpi):
    global _ENVIRON

    if specs_fontsize is None:
        specs_fontsize = fontsize
    if tick_fontsize is None:
        tick_fontsize = fontsize
    if label_fontsize is None:
        label_fontsize = fontsize
    if legend_fontsize is None:
        legend_fontsize = fontsize

    plt.rcParams.update({'axes.titleweight': 'normal'})  # 'bold' or 'normal'

    custom_params = {'axes.titlelocation': ('specs_position', specs_position),
                     'axes.labelweight': ('axes_labelweight', 'bold'),
                     'axes.titlesize': ('specs_fontsize', specs_fontsize),
                     'axes.labelsize': ('label_fontsize', label_fontsize),
                     'figure.dpi': ('dpi', dpi),
                     'legend.fontsize': ('legend_fontsize', legend_fontsize),
                     'legend.title_fontsize': ('legend_title_fontsize', legend_fontsize),
                     'figure.constrained_layout.use': ('figure_constrained_layout_use', _ENVIRON == 'normal')}

    skip_names = ['legend.title_fontsize']
    for (i, (j0, j1)) in custom_params.items():
        if j1 is not None:
            if i in plt.rcParams:
                plt.rcParams.update({i: j1})
            elif i not in skip_names:
                msg = 'Attribute ' + i + ' is not found in matplotlib rcParams. Hence, ' + j0 + ' is not set.'
                msg += ' Using default rcParams value.'
                warnings.warn(msg, stacklevel=2)
    return tick_fontsize


def _assign_mplibrcparams_for_subplot(fontsize,
                                      label_fontsize,
                                      title_fontsize,
                                      legend_fontsize,
                                      tick_fontsize,
                                      title_position,
                                      specs_fontsize,
                                      dpi):
    if dpi is not None:
        plt.rcParams['figure.dpi'] = dpi

    if specs_fontsize is None:
        specs_fontsize = fontsize

    if label_fontsize is None:
        label_fontsize = fontsize

    if title_fontsize is None:
        title_fontsize = fontsize

    if tick_fontsize is None:
        tick_fontsize = fontsize

    if legend_fontsize is None:
        legend_fontsize = fontsize

    # Set plt.rcParams
    plt.rcParams['axes.labelsize'] = label_fontsize
    plt.rcParams.update({'axes.titlelocation': title_position})
    plt.rcParams.update({'axes.titlesize': title_fontsize})
    plt.rcParams.update({'legend.title_fontsize': legend_fontsize})
    plt.rcParams.update({'legend.fontsize': legend_fontsize})

    # Non-parametric ones
    plt.rcParams.update({'axes.titleweight': 'bold'})  # 'bold' or 'normal'
    plt.rcParams.update({'axes.labelweight': 'bold'})  # 'bold' or 'normal'

    return tick_fontsize, specs_fontsize


def _getxticks(df, set_xticks_from_index=True):
    lbl = df.index.values
    d = np.array(lbl).dtype
    is_xticks_number = np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer)

    xticks = None
    if set_xticks_from_index:
        if is_xticks_number:
            xticks = df.index
        else:
            xticks = range(len(df))
    return is_xticks_number, xticks


def _get_groupings_old(df, subplot_argID):
    names = pd.DataFrame(np.array(list(df.index), dtype=object), columns=df.index.names)
    c = names.columns

    if (subplot_argID >= len(c)) or (subplot_argID < -len(c)):
        raise Exception('subplot_argID exceeds number of functions available.')

    cols = np.setdiff1d(c, c[subplot_argID])
    p = names.groupby(cols.tolist()).indices

    nums = np.array(list(map(len, p.values())))
    groupings_done = (nums[0] == nums).all() & (nums[0] > 1)
    # print('groupings_done : '+str(groupings_done))

    if not groupings_done:
        warnings.warn('Groupings not done', stacklevel=2)

    out, ncols = None, None
    if groupings_done:
        out = {}
        for k, v in p.items():
            df0 = df.iloc[v]
            df0.index = df0.index.get_level_values(c[subplot_argID])
            out[k] = df0

        ncols = len(names.drop(names.columns[subplot_argID], axis=1).iloc[:, -1].unique())
    return groupings_done, ncols, out


def _get_groupings(df, subplot_argID):
    # Default output values
    out, ncols = None, None

    names = pd.DataFrame(np.array(list(df.index), dtype=object), columns=df.index.names)
    c = names.columns

    if (subplot_argID >= len(c)) or (subplot_argID < -len(c)):
        raise Exception('subplot_argID exceeds number of functions available.')

    cols = np.setdiff1d(c, c[subplot_argID])

    p = names.groupby(cols.tolist()).groups

    first_grp_index_vals = names.iloc[p[list(p.keys())[0]], subplot_argID].values

    # Check for absolute index matches, thus absolute combinations only.
    #### FW : For future work, we might want to extend to partial combinations.
    equal_indexes = all([np.array_equal(names.iloc[v, subplot_argID].values, first_grp_index_vals) for (k, v) in p.items()])
    groupings_done = equal_indexes and len(first_grp_index_vals) > 1
    # print('groupings_done : '+str(groupings_done))

    if not groupings_done:
        warnings.warn('Groupings not done', stacklevel=2)

    if not groupings_done:
        return groupings_done, ncols, out
    else:
        out = {}
        for k, v in p.items():
            df0 = df.iloc[v]
            df0.index = df0.index.get_level_values(c[subplot_argID])
            out[k] = df0

        ncols = len(names.drop(names.columns[subplot_argID], axis=1).iloc[:, -1].unique())
        return groupings_done, ncols, out


def _mapvals(val, map_dict, name=None):
    if name is None:
        err_msg = 'Wrong value.'
    else:
        err_msg = 'Wrong value for ' + name + '.'
    if val not in map_dict:
        raise ValueError(err_msg)
    return map_dict[val]


# Setup title str for each subplot
def _subplot_title_str(df, argID):
    list_names = list(df.index.names)
    list_names.pop(argID)
    if len(list_names) == 1:
        title_str = list_names[0] + ' : '
    else:
        title_str = '(' + ', '.join(list_names) + ') : '
    return title_str


def bench(df, dtype='t', copy=False, multivar=False, multiindex=False):
    """
    Constructor function for creating BenchmarkObj object from a pandas dataframe.
    With input arguments, it could set as a timings or speedups or scaled-timings object.
    Additionally, the dataframe could be copied so that source dataframe stays unaffected.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe listing the timings or speedups or scaled-timings or just any 2D data,
        i.e. number of levels with rows and columns is 1.
        Also, the dataframe should have the benchmarking information setup in the standardized setup way.
        Columns represent function names, alongwith df.columns.name assigned as 'Functions'.
        Index values represent dataset IDs, alongwith df.index.name assigned based on dataset type.
    dtype : str, optional
        Datatype value that decides between timings or speedups or scaled-timings.
        Mapping strings are : 't' -> timings, 'st' -> scaled-timings, 's' -> speedups.
    copy : bool, optional
        Decides whether to copy data when constructing benchamrking object.

    Returns
    -------
    BenchmarkObj
        Data stored in BenchmarkObj.

    """

    map_dtype = {'t': 'timings', 's': 'speedups', 'st': 'scaled_timings'}
    if dtype in map_dtype:
        dt = map_dtype[dtype]
    elif dtype in map_dtype.values():
        dt = dtype
    else:
        raise TypeError('data type ' + '"' + str(dtype) + '" not understood')

    if copy:
        df_new = df.copy()
    else:
        df_new = df

    return BenchmarkObj(df_new, dtype=dt, multivar=multivar, multiindex=multiindex)


def _xticks_info_from_df(df):
    xticklabels = df.index

    idx = np.array(df.index)
    d = idx.dtype
    if np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer):
        xticks = df.index
    else:
        warnings.warn("Invalid index for use as plot xticks. Using range as the default xticks.")
        xticks = range(len(df))

    return xticklabels, xticks


def _get_plotparams_from_obj(df, dtype, logx, logy, ylabel):
    if ylabel is None:
        ylabel_map = {'timings': 'Runtime (s)', 'speedups': 'Speedup (x)', 'scaled_timings': 'Scaled Runtime (x)'}
        ylabel = ylabel_map[dtype]

    if logy is None:
        logy_map = {'timings': True, 'speedups': False, 'scaled_timings': False}
        logy = logy_map[dtype]

    if len(df) == 1 and logx:
        logx = False
        warnings.warn("Length of input dataframe is 1. Forcing it to linear scale for logx.")

    xticklabels, xticks = _xticks_info_from_df(df)
    return logx, logy, ylabel, xticklabels, xticks


class BenchmarkObj(object):
    """
    Class that holds various methods to benchmark solutions on various aspects of benchmarking metrics.
    This also includes timing and plotting methods. The basic building block is a pandas dataframe that
    lists timings off various methods. The index has the various datasets and headers are functions.
    This class is intended to hold timings data. It is the central building block to benchmarking workflow..
    """

    def __init__(self, df_timings, dtype='timings', multivar=False, multiindex=False):
        self.__df_timings = df_timings
        self.dtype = dtype
        self.multivar = multivar
        self.multiindex = multiindex
        self.__cols = df_timings.columns

    def scaled_timings(self, ref):
        """
        Evaluate scaled timings for all function calls with respect to one among them.

        Parameters
        ----------
        ref : int or str or function
            Input value represents one of the headers in the input BenchmarkObj.
            The scaled timings for all function calls are computed with respect to
            this reference.

        Returns
        -------
        BenchmarkObj
            Scaled timings.
        """

        if self.dtype != 'timings':
            raise AttributeError('scaled_timings is not applicable on ' + self.dtype + ' object')

        # Get timings dataframe
        df = self.__df_timings

        # Compute reference index
        if isinstance(ref, int):
            index = ref
        elif type(ref) is str:
            if ref not in df.columns:
                raise ValueError("Invalid ref function string.")
            index = df.columns.get_loc(ref)
        elif hasattr(ref, '__name__'):
            if ref.__name__ not in df.columns:
                raise ValueError("Invalid ref function.")
            index = df.columns.get_loc(ref.__name__)
        else:
            raise ValueError("Invalid ref value. Please check docs for valid ones.")

        # Get scaled dataframe and hence the new BenchmarkObj
        st = df.div(df.iloc[:, index], axis=0)
        st.rename({st.columns[index]: 'Ref:' + st.columns[index]}, axis=1, inplace=True)
        return BenchmarkObj(st, dtype='scaled_timings', multivar=self.multivar, multiindex=self.multiindex)

    def speedups(self, ref):
        """
        Evaluate speedups for all function calls with respect to one among them.

        Parameters
        ----------
        ref : int or str or function
            Same as with scaled_timings.

        Returns
        -------
        BenchmarkObj
            Speedups.
        """

        if self.dtype != 'timings':
            raise AttributeError('speedups is not applicable on ' + self.dtype + ' object')

        s = 1. / BenchmarkObj.scaled_timings(self, ref).to_dataframe()
        return BenchmarkObj(s, dtype='speedups', multivar=self.multivar, multiindex=self.multiindex)

    def drop(self, labels, axis=1):
        """
        Drop functions or datasets off the benchmarking object based on column or index values.
        It is an in-place operation.

        Parameters
        ----------
        labels : Any scalar or list or tuple of scalars
            Column or index value(s) to be dropped.

        Returns
        -------
        None
            NA.
        """

        df = self.__df_timings
        self.__df_timings = df.drop(labels, axis=axis)
        return

    def show_columns(self):
        """
        Get reference to inherent dataframe columns.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.core.indexes.base.Index
            Array of inherent dataframe columns.
        """

        df = self.__df_timings
        return df.columns

    def show_index(self):
        """
        Get reference to inherent dataframe columns.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.core.indexes.base.Index
            Array of inherent dataframe columns.
        """

        df = self.__df_timings
        return df.index

    def rank(self, mode='range'):
        """
        Rank different functions based on their performance number and rank them by
        changing the columns order accordingly. It is an in-place operation.

        Parameters
        ----------
        mode : str, optional
            Sets the ranking criteria to rank different functions.
            It must be one among - 'range', 'constant', 'index'.

        Returns
        -------
        None
            NA.
        """

        df = self.__df_timings
        multiindex = self.multiindex

        if multiindex:
            raise Exception('Ranking is supported for groupings.')

        if mode == 'range':
            R = np.arange(1, len(df) + 1)
        elif mode == 'constant':
            R = np.ones(len(df))
        elif mode == 'index':
            idx = np.array(df.index)
            d = idx.dtype
            if np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer):
                R = df.index.values
            else:
                raise ValueError("Dataframe index is not int or float. Hence, 'index' is an invalid option as mode.")
        else:
            raise ValueError("Invalid option as mode.")

        df_ranked = df.iloc[:, R.dot(df).argsort()[::-1]]
        df[:] = df_ranked
        df.columns = df_ranked.columns
        return

    def copy(self):
        """
        Make a copy.

        Returns
        -------
        BenchmarkObj
            Copy of input BenchmarkObj object.
        """

        return BenchmarkObj(self.__df_timings.copy(), dtype=self.dtype, multivar=self.multivar, multiindex=self.multiindex)

    def plot(self, set_xticks_from_index=True,
             xlabel=None,
             ylabel=None,
             colormap='jet',
             logx=False,
             logy=None,
             grid=True,
             linewidth=2,
             rot=None,
             dpi=None,
             fontsize=14,
             specs_fontsize=None,
             tick_fontsize=None,
             label_fontsize=None,
             legend_fontsize=None,
             figsize=None,
             specs_position='left',
             debug_plotfs=False,
             pause_timefs=0.1,
             modules=None,
             use_frame=False,
             sp_argID=0,
             sp_ncols=-1,
             sp_sharey=None,
             sp_title_position='center',
             sp_title_fontsize=None,
             sp_show_specs=True,
             save=None,
             **kwargs):
        """
        Plot dataframe using given input parameters.

        Parameters
        ----------
        set_xticks_from_index : bool, optional
            Flag to use dataframe's index to set set_xticklabels or not.
        xlabel : str, optional
            Xlabel string.
        ylabel : str, optional
            Ylabel string.
        colormap : str, optional
            String that decides the colormap for plotting
        logx : bool, optional
            Flag to set x-axis scale as log or linear.
        logy : None or bool, optional
            If set as None, it detects default boolean flag using input Object datatype
            to be used as logy argument for plotting that decides the y-axis scale.
            With True and False, the scale is log and linear respectively.
            If set as boolean, it is used directly as logy argument.
        grid : bool, optional
            Flag to show grid or not.
        linewidth : int, optional
            Width of line to be used for plotting.
        rot : int or None, optional
            Rotation for ticks (xticks for vertical, yticks for horizontal plots).
        dpi : float or None, optional
            The resolution of the figure in dots-per-inch.
        fontsize : float or int or None, optional
            Fontsize used across specs_fontsize, tick_fontsize and label_fontsize if they are not set.
        specs_fontsize : float or int or None, optional
            Fontsize for specifications text displayed as title.
        tick_fontsize : float or int or None, optional
            Fontsize for xticks and yticks.
        label_fontsize : float or int or None, optional
            Fontsize for xlabel and ylabel.
        figsize : tuple of two integers or None, optional
            Tuple with syntax (figure_width, figure_height) for the figure window.
            This is applied only for environemnts where full-screen viewing is not possible.
        specs_position : None or str, optional
            str that decides where to print specs information.
            Options are : None(default), 'left', 'right' and 'center'.
        debug_plotfs : bool, optional
            Flag to decide whether to display debug info on fullscreen showing of plot.
            This is used only for interactive backends.
        pause_timefs : float, optional
            This is a pause number in seconds, used for plot to be rendered in fullscreen before saving it.
        modules : dict, optional
            Dictionary of modules.
        use_frame : bool, optional
            This indicates whether to use a frame or not.
            For subplot, this applies a frame to each subplot.
        sp_argID : int, optional
            This is specific to subplot case, when applicable (combinations are possible).
            This represents argument index for the input datasets to be used as the base (for x-axis labelling).
            This is based on 0-based indexing. Default argument index is 0, i.e. the first argument.
        sp_ncols : int, optional
            This is specific to subplot case, when applicable (combinations are possible).
            This denotes the number of columns used to create subplot grid.
        sp_sharey : str or None, optional
            This is specific to subplot case, when applicable (combinations are possible).
            This is used to indicate if and how the y-values are to be shared.
            Accepted values and their respective functionalities are listed below :
                None : y-values are not shared.
                'row' or 'r': y-values are shared among same row of subplots.
                'global' or 'g': y-values are shared globally across all subplots.
        sp_title_position : str, optional
            This is specific to subplot case, when applicable (combinations are possible).
            This indicates where to place the title for each subplot.
            Available values are - 'left', 'center' or 'right' respective to their positions.
        sp_title_fontsize : float or int or None, optional
            This is specific to subplot case, when applicable (combinations are possible).
            Fontsize for title for subplots that shows the grouping argument(s).
        sp_show_specs : bool, optional
            This decides whether to show specifications or not.
            Default is True, i.e show specifications.
        save : str or None, optional
            Path to save plot.
        **kwargs
            Options to pass to pandas plot method, including kwargs for matplotlib plotting method.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot of data from object's dataframe.

        Notes
        -----
        All subplot specific arguments have prefix of "sp_".
        """

        # Get dataframe and dtype
        dtype = self.dtype
        df = self.__df_timings
        multiindex = self.multiindex

        style.use('bmh' if use_frame else 'fivethirtyeight')

        # Setup styles
        available_linestyles = ['-.', '--', '-']
        extls = np.resize(available_linestyles, df.shape[1]).tolist()

        # Get plot params - logy, ylabel
        if ylabel is None:
            ylabel_map = {'timings': 'Runtime [s]', 'speedups': 'Speedup [x]', 'scaled_timings': 'Scaled Runtime [x]'}
            ylabel = ylabel_map[dtype]

        if logy is None:
            logy_map = {'timings': True, 'speedups': False, 'scaled_timings': False}
            logy = logy_map[dtype]

        groupings_done = False
        if multiindex:
            # Get groupings
            groupings_done, ncols, out = _get_groupings(df, sp_argID)
            if not groupings_done:
                warnings.warn('It is multivar, but groupings are not possible. Hence, resorting to normal plot.', stacklevel=2)

        if not groupings_done:  # normal plot
            is_xticks_number, xticks = _getxticks(df, set_xticks_from_index=set_xticks_from_index)

            if not is_xticks_number and logx:
                xticks = None
                warnings.warn('Some xticks might be not be seen.', stacklevel=2)

            tick_fontsize = _assign_mplibrcparams(fontsize, specs_fontsize, tick_fontsize,
                                                  label_fontsize, legend_fontsize, specs_position, dpi)

            # Plot using dataframe data and its attributes
            ax = df.plot(style=extls,
                         colormap=_truncate_cmap(colormap),
                         title=specs(modules=modules),
                         rot=rot,
                         fontsize=tick_fontsize,
                         linewidth=linewidth,
                         logx=logx,
                         logy=logy,
                         figsize=figsize,
                         xticks=xticks,
                         **kwargs)

            if grid:
                ax.grid(True, which="both", ls="-")
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if set_xticks_from_index and is_xticks_number:
                ax.set_xticklabels(df.index)
            if not is_xticks_number:  # earlier set_xticks_from_index or not is_xticks_number
                ax.set_xticks([], minor=True)

            # Show in fullscreen
            fullscreenfig(ax, pause_timefs, print_info=debug_plotfs)
            ax_fig = ax.figure
        else:  # subplot
            _SUBPLOT_FIGSIZE = (6.4, 3.6)

            if figsize is None:
                figsize = _SUBPLOT_FIGSIZE

            if sp_ncols == -1:
                sp_ncols = ncols

            tick_fontsize, specs_fontsize = _assign_mplibrcparams_for_subplot(fontsize, label_fontsize, sp_title_fontsize,
                                                                              legend_fontsize, tick_fontsize, sp_title_position,
                                                                              specs_fontsize, dpi)

            len1 = len(out)
            nrows = int(np.ceil(len1 / float(sp_ncols)))
            r, c = np.unravel_index(np.arange(len1), (nrows, sp_ncols))

            map_dict = {None: False, 'row': 'row', 'r': 'row', 'global': True, 'g': True}
            sharey_val = _mapvals(sp_sharey, map_dict, name='subplot_sharey')

            # Setup title str for each subplot
            title_str = _subplot_title_str(df, sp_argID)

            df0 = out[list(out.keys())[0]]
            is_xticks_number, xticks = _getxticks(df0, set_xticks_from_index=set_xticks_from_index)

            if not is_xticks_number and logx:
                xticks = None
                warnings.warn('Some xticks might be not be seen.', stacklevel=2)

            figS = (sp_ncols * figsize[0], nrows * figsize[1])
            fig, axs = plt.subplots(nrows, sp_ncols, sharex='col', constrained_layout=True, sharey=sharey_val, figsize=figS)
            axs = axs.reshape(-1, sp_ncols)

            for i, (k, df0) in enumerate(out.items()):
                ri, ci = r[i], c[i]
                ax = axs[ri, ci]
                is_first_subplot = (ri == 0) and (ci == 0)

                subplot_title = str(k)
                if is_first_subplot:
                    subplot_title = title_str + subplot_title

                df0.plot(ax=ax,
                         style=extls,
                         linewidth=linewidth,
                         colormap=_truncate_cmap(colormap),
                         xticks=xticks,
                         logx=logx,
                         logy=logy,
                         rot=rot,
                         title=subplot_title,
                         fontsize=tick_fontsize,
                         legend=i == 0,
                         **kwargs)

                if grid:
                    ax.grid(True, which="both", ls="-")

                # Next two IFs are needed for logx=True plots
                if set_xticks_from_index and is_xticks_number:
                    ax.set_xticklabels(df0.index)
                if not is_xticks_number:  # earlier set_xticks_from_index or not is_xticks_number
                    ax.set_xticks([], minor=True)

            # Set ylabel on middle row ID
            axs[nrows // 2, 0].set_ylabel(ylabel)

            if sp_show_specs:
                fig.suptitle(specs_short(), fontsize=specs_fontsize)

            # remove unused axes
            xlabel = df.index.names[sp_argID]
            r, c = np.unravel_index(np.arange(len1, axs.size), (nrows, sp_ncols))
            for (i, j) in zip(r, c):
                ax = axs[i, j]
                ax.set_xlabel(xlabel)
                ax.tick_params(labelrotation=rot)
                ax.yaxis.set_visible(False)
                ax.patch.set_visible(False)
                plt.setp(ax.spines.values(), visible=False)
            ax_fig = fig

        # Save figure
        if save is not None:
            ax_fig.savefig(save, bbox_inches='tight')
        return ax_fig

    def reset_columns(self):
        """Reset columns to original order."""

        reset_cols = [i for i in self.__cols if i in self.__df_timings.columns]
        self.__df_timings = self.__df_timings.loc[:, reset_cols]
        return

    def to_dataframe(self, copy=False):
        """Return underlying pandas dataframe object."""

        if not copy:
            return self.__df_timings
        else:
            return self.__df_timings.copy()

    def __str__(self):
        return repr(self.__df_timings)

    __repr__ = __str__
