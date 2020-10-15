Benchmarking steps
==================

A minimal workflow employing this package would basically involve three steps :

* Setup functions : A list or dictionary of functions to be benchmarked. It supports both single and multiple arguments.
* Setup datasets : A list or dictionary of datasets to be benchmarked.
* Benchmark to get timings in a dataframe-like object. Each row holds one dataset and each header represents one function each. Dataframe has been the design choice, as it supports plotting directly from it and additionally benchmarking setup information could be stored as name values for index and columns.

We will study these with the help of a sample setup in :ref:`Minimal workflow`.

We will study about setting up functions and datasets in detail later in this document.

.. note::

  Prior to Python 3.6, dictionary keys are not maintained in the order they are inserted. So, when working with those versions and with input dataset being defined as a dictionary, to keep the order, `collections.OrderedDict <https://docs.python.org/2/library/collections.html#collections.OrderedDict>`__ could be used.

To get more out of it, we could optionally do the following :

* Plot the timings.
* Get speedups or scaled-timings of all functions with respect to one among them.
* Rank the functions based on various performance-metrics.

A detailed study with examples in the next section should clear up things.

We will try to take a hands-on approach and explore the features available with this package. We will start off with the minimal steps to benchmarking on a setup and then explore other utilities to cover most common features.

Rest of the documentation will use the module's methods. So, let's import it once :

.. code-block:: python

    >>> import benchit

Minimal workflow
----------------

We will study a case of single argument with default parameters. Let's take a sample case where we try to benchmark the five most common NumPy ufuncs - `sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`__, `prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`__, `max <https://numpy.org/doc/stable/reference/genebenchrated/numpy.amax.html>`__, `mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`__, `median <https://numpy.org/doc/stable/reference/generated/numpy.median.html>`__ on arrays varying in their sizes. To keep it simple, let's consider `1D` arrays. Thus, the benchmarking steps would look something like this :

.. code-block:: python

    >>> import numpy as np
    >>> funcs = [np.sum,np.prod,np.max,np.mean,np.median]
    >>> inputs = [np.random.rand(i,i) for i in 4**np.arange(7)]
    >>> t = benchit.timings(funcs, inputs)
    >>> t
    Functions       sum      prod      amax      mean    median                                                                                                                                                        
    Len                                                                                                                                                                                                                
    1          0.000005  0.000004  0.000005  0.000007  0.000046
    4          0.000005  0.000004  0.000005  0.000007  0.000047
    16         0.000005  0.000005  0.000005  0.000007  0.000049
    64         0.000007  0.000014  0.000007  0.000009  0.000094
    256        0.000035  0.000131  0.000030  0.000038  0.000845
    1024       0.000511  0.002050  0.000512  0.000522  0.011525
    4096       0.008208  0.032582  0.008257  0.008274  0.261838

It's a *dataframe-like* object, called `BenchmarkObj`. We can plot it, which automatically adds in system configuration into the title area to convey all the available benchmarking information :

.. code-block:: python

    >>> t.plot(logy=True, logx=True, save='timings.png')

Resultant plot would look something like this :

|timings|


These `4` lines of codes would be enough for most of the benchmarking workflows.


Extract dataframe & construct back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The underlying benchmarking data is stored as a pandas dataframe that could be extracted with :

.. code-block:: python

    >>> df = t.to_dataframe()

As we shall see in the next sections, this would be useful in our benchmarking quest to extend the capabilities.

There's a benchmarking object construct function `benchit.bench` that accepts dataframe alongwith `dtype`. So, we can construct it, like so :

.. code-block:: python

    >>> t = benchit.bench(df, ...)


Setup functions
---------------

This would be a list or dictionary of functions to be benchmarked.

A general syntax for list version would look something like this :

.. code-block:: python

    >>> funcs = [func1, func2, ...]

We already saw a sample of it in :ref:`Minimal workflow`.

A general syntax for dictionary version would look something like this :

.. code-block:: python

    >>> funcs = {'func1_name':func1, 'func2_name':func2, ...}

Mixing in lambdas
^^^^^^^^^^^^^^^^^

`Lambda functions <https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions>`__ could also be mixed into our functions for benchmarking with a dictionary. So, the general syntax would be :

.. code-block:: python

    >>> funcs = {'func1_name':func1, 'lambda1_name':lamda1, 'func2_name':func2, ...}

This is useful for directly incorporating one-liner solutions without the need of defining them beforehand.

Let's take a sample setup where we will tile a `1D` array twice with various solutions as lambda and regular functions mixed in :

.. code-block:: python

    import numpy as np

    def numpy_concat(a):
        return np.concatenate([a, a])

    # We need a dictonary to give each lambda an unique name, through keys
    funcs = {'r_':lambda a:np.r_[a, a],
             'stack+reshape':lambda a:np.stack([a, a]).reshape(-1),
             'hstack':lambda a:np.hstack([a, a]),
             'concat':numpy_concat,
             'tile':lambda a:np.tile(a,2)}


Setup datasets
--------------

This would be a list or dictionary of datasets to be benchmarked.

A general syntax for list version would look something like this :

.. code-block:: python

    >>> in_ = [dataset1, dataset2, ...]

For such list type `inputs`, based on the datasets and additional argument `indexby` to `benchit.timings`, each dataset is assigned an `index`.

A general syntax for dictionary version would look something like this :

.. code-block:: python

    >>> in_ = {'argument_value1':dataset1, 'argument_value2':dataset2, ...}

For such dictionary type `inputs`, index values would be the dictionary keys.

For both lists and dicts, these index values are used for plotting, etc. With single argument cases, this is pretty straight-forward.

Now, we might have functions that accept more than one argument, let's call those as `multivar` cases and focus on those. Please keep in mind that for those `multivar` cases, we need to feed in `multivar=True` into `benchit.timings`.

Pseudo code would look something like this :

.. code-block:: python

    >>> in_ = {m:generate_inputs(m,k1,k2) for m in m_list} # k1, k2 are constants
    >>> t = benchit.timings(fncs, in_, multivar=True, input_name='arg0')


Multiple arguments
^^^^^^^^^^^^^^^^^^

With some of those `multivar` cases, we might want to use keys as tuples or lists with a string each for each of the argument to the input as better representatives for each of the datasets. These would help us with plotting among others, as we shall see later.

Thus, with functions that accept two arguments, it would be :

.. code-block:: python

    >>> in_ = {('argument1_value1','argument2_value1'):dataset1,
               ('argument1_value2','argument2_value2'):dataset2, ...}

Groupings
"""""""""

A specific case of forming such tuple/list key based dictionaries, we would be with nested loops. Such a setup enables us to form groupings, let's call them `multivar-groupings`. A sample one would look something like this :

.. code-block:: python

    >>> in_ = {('argument1_value1','argument2_value1'):dataset1,
               ('argument1_value1','argument2_value2'):dataset2,
               ('argument1_value1','argument2_value3'):dataset3,
               ('argument1_value2','argument2_value1'):dataset4,
               ('argument1_value2','argument2_value2'):dataset5,
               ('argument1_value2','argument2_value3'):dataset6, ...}

Regardless of the way `inputs` is setup, `benchit` would try to form combinations.

So, for the `6` datasets case :

- Considering `argument1` values as reference, we would have `2` groups - `(dataset1, 2, 3)` and `(dataset4, 5, 6)`.
- Considering `argument2` values as reference, we would have `3` groups - `(dataset1, 4)`,  `(dataset2, 5)` and `(dataset3, 6)`.

Then, those groupings could be plotted as subplots.

Optionally, to finalize the groupings with proper names, we can assign names to each argument with `input_name` argument to `benchit.timings`. So, `input_name` would be a list or tuple specifying the names for each argument as its elements as strings. These would be picked up for labelling purpose when plotting.


Pseudo code would look something like this :

.. code-block:: python

    >>> in_ = {(m,n):generate_inputs(m,n) for m in m_list for n in n_list}
    >>> t = benchit.timings(fncs, in_, multivar=True, input_name=['arg0', 'arg1'])

Plots on groupings would result in subplots. More on this with examples is shown later in this document.


.. |timings| image:: timings.png
