import numpy as np
import pandas as pd
import timeit
import functools
import warnings
from tqdm import trange
from benchit.plot_utils import _add_specs_as_title, _add_specs_as_textbox, _truncate_cmap

# Parameters
_TIMEOUT = 0.2
_NUM_REPEATS = 5


def _get_timings_perinput(funcs, input_=None):
    """
    Evaluates function calls on a given input to compute the timings.

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

    timings_l = []
    for j in trange(len(funcs), desc='Loop functions', leave=False):
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
    Evaluates function calls on given inputs to compute the timings.

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
    Given inputs in a list or dict gets the possible indexby arguments

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
        Given inputs in a list and indexby option, sets xticklabels and
        indexby string based on type and format of inputs.

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
            """Format array shape info"""
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
        Given inputs in a dict, sets xticklabels and indexby string based on
        type and format of inputs.

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
    Evaluates function calls on given input(s) to compute the timing.
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

    if input_name is not None:
        df_timings.index.name = input_name
    else:
        df_timings.index.name = xlabel_from_inputs

    benchObj = BenchmarkObj(df_timings)
    return benchObj

def bench(df, dtype='t', copy=False):
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
    
    map_dtype = {'t':'timings', 's':'speedups', 'st':'scaled_timings'}
    if dtype in map_dtype:
        dt = map_dtype[dtype]
    elif dtype in map_dtype.values():
        dt = dtype
    else:
        raise TypeError('data type ' + '"' + str(dtype) + '" not understood')        

    if not copy:
        return BenchmarkObj(df, dtype=dt)
    else:
        return BenchmarkObj(df.copy(), dtype=dt)
    

class BenchmarkObj(object):
    """
    Class that holds various methods to benchmark solutions on various aspects of benchmarking metrics.
    This also includes timing and plotting methods. The basic building block is a pandas dataframe that
    lists timings off various methods. The index has the various datasets and headers are functions.
    This class is intended to hold timings data. It is the central building block to benchmarking workflow..
    """

    def __init__(self, df_timings, dtype='timings'):
        self.__df_timings = df_timings
        self.dtype = dtype
        self.__cols = df_timings.columns

    def scaled_timings(self, index):
        """
        Evaluates scaled timings for all function calls with respect to one among them.

        Parameters
        ----------
        index : int
            Column ID of the reference function call in the input BenchmarkObj.
            The scaled timings for all function calls are computed with respect to
            this reference column ID.

        Returns
        -------
        BenchmarkObj
            Scaled timings.
        """
        
        if self.dtype != 'timings':
            raise AttributeError('scaled_timings is not applicable on '+self.dtype+' object')            

        st = self.__df_timings.div(self.__df_timings.iloc[:, index], axis=0)
        st.rename({st.columns[index]: 'Ref:' + st.columns[index]}, axis=1, inplace=True)
        stB = BenchmarkObj(st,'scaled_timings')
        return stB

    def speedups(self, index):
        """
        Evaluates speedups for all function calls with respect to one among them.

        Parameters
        ----------
        index : int
            Same as with scaled_timings.

        Returns
        -------
        BenchmarkObj
            Speedups.
        """
        
        if self.dtype != 'timings':
            raise AttributeError('speedups is not applicable on '+self.dtype+' object')            
        
        s = 1./BenchmarkObj.scaled_timings(self, index).to_dataframe()
        sB = BenchmarkObj(s,'speedups')
        return sB

    def drop(self, labels, axis):
        """
        Drops functions off the benchmarking object based on column index numbers.
        It is an in-place operation.

        Parameters
        ----------
        index : int or tuple/list of int column index value(s) to be dropped.

        Returns
        -------
        None
            NA.
        """
        df = self.__df_timings        
        self.__df_timings = df.drop(labels,axis=axis)
        return
                
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
        
        if mode == 'range':
            R = np.arange(1,len(df)+1)
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
    
        df_ranked = df.iloc[:,R.dot(df).argsort()[::-1]]
        df[:] = df_ranked
        df.columns = df_ranked.columns
        return

    def copy(self):
        """
        Makes a copy.        
            
        Returns
        -------
        BenchmarkObj
            Copy of input BenchmarkObj object.
        """
        
        return BenchmarkObj(self.__df_timings.copy(), self.dtype)

        
    def plot(self, set_xticks_from_index=True,
             xlabel=None,
             ylabel=None,
             colormap='jet',
             marker='',
             logx=False,
             logy=True,
             grid=True,
             linewidth=2,
             add_specs_as='title',
             modules=None,
             save=None):
        """
        Plots dataframe using given input parameters.
    
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
        marker : str, optional
            String that decides the markers for plotting.
        logx : bool, optional
            Flag to set x-axis scale as log or linear.
        logy : bool, optional
            Flag to set y-axis scale as log or linear.
        grid : bool, optional
            Flag to show grid or not.
        linewidth : int, optional
            Width of line to be used for plotting.
        add_specs_as : str, optional
            Decides the position to add specs information.
        modules : dict, optional
            Dictionary of modules.
        save : str or None, optional
            Path to save plot.
    
        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
            Plot of data from object's dataframe.
        """
    
        # Get dataframe and dtype        
        dtype = self.dtype                
        df = self.__df_timings  
        
        if ylabel is None:
            ylabel_map = {'timings':'Runtime (s)', 'speedups':'Speedup (x)', 'scaled_timings':'Scaled Runtime (x)',}
            ylabel = ylabel_map[dtype]        
    
        if len(df) == 1 and logx:
            logx = False
            warnings.warn("Length of input dataframe is 1. Forcing it to linear scale for logx.")
            
        available_linestyles = ['-.','--','-']
        extls = np.resize(available_linestyles, df.shape[1]).tolist()
        dfp = df.plot(style=extls, colormap=_truncate_cmap(colormap), logx=logx, logy=logy, linewidth=linewidth)
    
        if set_xticks_from_index:
            dfp.set_xticklabels(df.index)
    
            idx = np.array(df.index)
            d = idx.dtype
            if np.issubdtype(d, np.floating) or np.issubdtype(d, np.integer):
                dfp.set_xticks(df.index)
            else:
                warnings.warn("Invalid index for use as plot xticks. Using range as the default xticks.")
                dfp.set_xticks(range(len(df)))
    
        if grid:
            dfp.grid(True, which="both", ls="-")
        if xlabel is not None:
            dfp.set_xlabel(xlabel)
        if ylabel is not None:
            dfp.set_ylabel(ylabel)
    
        if add_specs_as == 'title':
            _add_specs_as_title(dfp, modules=modules)
        elif add_specs_as == 'textbox':
            _add_specs_as_textbox(dfp, modules=modules)
        else:
            raise ValueError("Must be a string with value 'title' or 'textbox'")
    
        # Save axes plot as an image file
        if save is not None:
            dfp.figure.savefig(save, bbox_inches='tight')
    
        return dfp

    def reset_columns(self):
        """Resets columns to original order."""
        self.__df_timings = self.__df_timings.loc[:,self.__cols]
        return

    def to_dataframe(self, copy=False):
        """Returns underlying pandas dataframe object."""

        if not copy:
            return self.__df_timings
        else:
            return self.__df_timings.copy()
    
    def __str__(self):
        return repr(self.__df_timings)

    __repr__ = __str__