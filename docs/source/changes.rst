Changelog
=========

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
