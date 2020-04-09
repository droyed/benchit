Introduction
===================================

We often end up with more than one way to solve a problem and at times we need to compare them based on certain criteria, which could be memory-efficiency or performance. Comparative analysis is an essential process to evaluate different methods on those criteria. Usually the problem setup involves various datasets that in some way represent various possible intended use-cases. Such a problem setup helps us present an in-depth analysis of the available methods across those cases. Please note that with this package, we are solely focusing on benchmarking pertaining to Python.

Relevant scenarios
------------------

Many times we use different Python modules to solve a problem. Python modules like NumPy, Numba, SciPy, etc. are built on different philosophies and hence fair differently on different datasets. Often one of the requirements when evaluating solutions with them or even with Vanilla Python becomes runtime performance. With this package, we are primarily focusing on evaluating runtime performance with different methods across different datasets.

The benchmarking process should cover all Python supported data, but the main motivation with this package has been to perform benchmarking on NumPy ndarrays and Pandas dataframe, Python lists and scalars.