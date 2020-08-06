#!/bin/bash

cd main_testing/
python3 testfordocs_readme.py    
python3 testfordocs_index_codes.py 
python3 testfordocs_realistic_codes.py
python3 testfordocs_workflow_codes.py 

exit 0