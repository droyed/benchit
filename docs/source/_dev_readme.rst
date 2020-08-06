Local installation
------------------

cd to path that has `benchit`'s `setup.py`.

Run on system console :

.. code-block:: bash

    $ sudo python -m pip install .
    $ sudo python3 -m pip install .

Before release
--------------

Change release tags in :

* benchit/__init__.py
* docs/source/conf.py

Confirm with (example for 0.0.4 release) :

.. code-block:: bash

    $ grep -rn "0\.0\.4"  --include \*.py
    benchit/__init__.py:4:__version__ = '0.0.4'
    docs/source/conf.py:28:release = u'0.0.4'
