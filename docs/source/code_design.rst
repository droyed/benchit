Code Design
============

The codebase is built upon the following philosophies -

* Input functions and datasets could be fed in as list or dictionary. It supports both single and multiple arguments. For multiple argument cases, we can pack the arguments in a list or tuple and thus each element from it becomes one argument each to be fed into a function for evaluation.

* Code structure allows end-user to write minimal code and get the benchmarking numbers and plots, as the parameters are picked up from the given inputs. Also, included benchmarking object holds all of the benchmarking setup information alongwith the timing numbers in a pandas dataframe-like object. The object has plot method to plot those numbers alongwith the setup information being shown in the same. The idea is to minimize coding efforts to share benchmarked results with maximum possible info.

Examples in the next section should help clarify on these points.