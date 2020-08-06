Benchmarking steps
==================

A minimal workflow employing this package would basically involve three steps :

* Setup functions : A list or dictionary of functions to be benchmarked. It supports both single and multiple arguments.
* Setup datasets : A list or dictionary of datasets to be benchmarked.
* Benchmark to get timings in a dataframe-like object. Each row holds one dataset and each header represents one function each. Dataframe has been the design choice, as it supports plotting directly from it and additionally benchmarking setup information could be stored as name values for index and columns.

We will study these with the help of a sample setup in :ref:`Minimal workflow`.

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

We will study a case of single argument with default parameters. Let's take a sample case where we try to benchmark the five most common NumPy ufuncs - `sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`__, `prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`__, `max <https://numpy.org/doc/stable/reference/generated/numpy.amax.html>`__, `mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`__, `median <https://numpy.org/doc/stable/reference/generated/numpy.median.html>`__ on arrays varying in their sizes. To keep it simple, let's consider `1D` arrays. Thus, the benchmarking steps would look something like this :

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


Mixing in lambdas
-----------------

`Lambda functions <https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions>`__ could be mixed into our functions for benchmarking with a dictionary. This is useful for directly incorporating one-liner solutions without the need of defining them beforehand. Let's take a sample setup where we will tile a `1D` array twice with various solutions as lambda and regular functions mixed in :

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


Thus, this `funcs` could be then be used to benchmark with `benchit.timings`.


Extract dataframe & construct back
----------------------------------

The underlying benchmarking data is stored as a pandas dataframe that could be extracted with :

.. code-block:: python

    >>> df = t.to_dataframe()

As we shall see in the next sections, this would be useful in our benchmarking quest to extend the capabilities.

There's a benchmarking object construct function `benchit.bench` that accepts dataframe alongwith `dtype`. So, we can do the constructing step in two ways :

.. code-block:: python

    >>> t = benchit.bench(df, dtype=t.dtype)



.. |timings| image:: timings.png
