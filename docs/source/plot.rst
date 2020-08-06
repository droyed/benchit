Plotting schemes
================

This is a brief discussion on various plotting schemes and tips that would be helpful in consideration, while customizing plots and plotting in different environments.

Plot features
-------------

For most of the plotting purposes, we can stick to `benchit`'s `plot` method - `benchit.BenchmarkObj.plot <https://benchit.readthedocs.io/en/latest/benchit.html#benchit.BenchmarkObj.plot>`__. This method enables `kwargs` to `pandas.DataFrame.plot <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.html#pandas-dataframe-plot>`__ Also, `pandas.DataFrame.plot` has its own `kwargs` that traces back to `matplotlib.pyplot.plot <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot>`__. In essence, within `benchit`'s `plot` we can explore all plot arguments available to `pandas` and `matplotlib` plot versions. This should be sufficient for most plotting requirements.

To go the full hog, we can employ two more methods, which could be used individually or in combination.

Method #1 : Modify matplotlib rc settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the `matplotlib` plot settings are stored in a dictionary-like variable called matplotlib.rcParams, which is global to the matplotlib package. More info on this is available at  - `Customizing Matplotlib with style sheets and rcParams <https://matplotlib.org/tutorials/introductory/customizing.html>`__. We can modify these to suit our plotting requirements. This setup is to be done before plotting.

Method #2 : Use axes methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`benchit`'s `plot` method returns an object of class `matplotlib.axes.Axes`. This has methods to change certain plot parameters. These could be located at `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes>`__. Most of those would be named as `Axes.set_[property]`. This is an after-plot adjustment and only applicable on interactive matplotlib backends.


Notebook plots
--------------

Plotting in IPython notebooks or Jupyter notebooks is supported for different `matplotlib` backends. Simply tell `benchit` to set the environment accordingly before plotting, with :

.. code-block:: python

    benchit.set_environ('notebook')


Note that this could also be used for non-interactive backends for better visualization. `Matplotlib backends <https://matplotlib.org/faq/usage_faq.html#what-is-a-backend>`__ lists these backends and provides some general information on backends.


`Sample notebook run <https://github.com/droyed/benchit/blob/master/docs/source/PlotDemo-NotebookEnv.ipynb>`__.


Plot tips
---------

When plotting with `benchit.BenchmarkObj.plot`, following tips could come in handy :

* If `xticks` seem congested, we can pass over the setting up for them to pandas version with `set_xticks_from_index` set as `False`. Another way would be to rotate `xticks` using its `rot` argument.

