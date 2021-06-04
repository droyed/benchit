Changelog
=========

0.0.6 (2021-06-04)
------------------

Changes :

- Added single-var groupings.
- Added props method to `BenchmarkObj` to list meta information about it.

Bug fixes :

- Fixed `ipython` dependency.


0.0.5 (2020-10-15)
------------------

Changes :

- Changes were applied at various levels to accommodate multiple arguments input with combinations cases mostly generated through nested loops over those arguments. Such combinations are led to generate subplots. This sub-plotting workflow is integrated to usual plotting mechanism to provide a one-stop solution. As such, the interface to the user stays the same, but with additional feature of sub-plotting for the combinations case.
- New one-line `specs` information.
- Framing feature added for plotting.
- Now we can quickly check out the layout and a general plot trend with a new function named `setparams`. This also replaces old function named `set_environ`, as we have few more arguments added to it.

Bug fixes :

- Fix for `logx` set as `True` and `set_xticks_from_index` set as `True` when dealing with various inputs on plotting.


0.0.4 (2020-08-06)
------------------

Changes :

- Progress bar fix for IPython/notebook runs to limit them to loop for datasets only.
- Documentation re-organized to show the minimal workflow and features separately.
- Introduced `set_environ` to set parameters for plotting, etc. according to `matplotlib` backend.
- Plot code reorganised to use specifications as title at dataframe plot level. This has helped in having an universal code to work for all backends. Makes use of `matplotlib` `rcparams` to setup environment before invoking dataframe plot method. The inspiration has been with notebook plotting. This has led to code cleanup to push more work in `main.py`.
- Owing to previous change, now deprecated `_add_specs_as_title`, `_add_specs_as_textbox`.


0.0.3 (2020-07-05)
------------------

Focus has been to make plots work across different matplotlib backends and few other plot improvements.

Changes :

- Added documentation support for lambdas.
- Added matplotlib inline plot support.
- Added different modes of argument for `speedups` and `scaled_timings` functions.
- Set customized default `logy` values based on `BenchmarkObj` datatypes.
- Added support for matplotlib backends other than `Qt5Agg` including notebook and inlining cases. Fix included a generic exception handling process to get fullscreen plots as is needed to put specifications as plot titles.

Bug fixes :

- `reset_columns` after `drop` works.

0.0.2 (2020-05-19)
------------------

With this release the focus has been to move the workflow from a tools perspective to a platform one. There's work done on a much tighter integration with `pandas-dataframe` format that should help to keep things on a platform-specific workflow.

Added methods to `BenchmarkObj` :

- `rank` to rank data based on certain performance metrics. This helps on making easier conclusions off the plots.
- `reset_columns` to retrieve original columns order. Intended to be used in conjunction with rank method.
- `copy` to retrieve original benchmarking results. This is to be used in cases where we might need to play around with the results and need a backup.
- `drop` to drop some functions or datasets to handle scenarios when we might want to focus on fewer ones.

Other changes :

- Added constructor for `BenchmarkObj` as `bench`.
- Renamed method `to_pandas_dataframe` for `BenchmarkObj` to `to_dataframe`.
- Nested calls to `speedups` and `scaled_timings` for `BenchmarkObj` disabled that are not applicable.
- Plot takes on automatic `ylabel` based on benchmarking object datatype.
- Introduced truncated colormap to avoid lighter colors, given the lighter background with the existing plotting scheme.
- Documentation pages re-arranged and improved to follow a workflow-based documentation.
- Renamed `bench.py` to `main.py` to avoid any conflicts with the new constructor to `BenchmarkObj` called `bench`.

0.0.1 (2020-04-10)
------------------

- Initial release.
