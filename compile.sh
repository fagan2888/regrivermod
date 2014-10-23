#!/bin/bash
cd regrivermod
python setup.py build_ext --inplace
cd ..
cd econlearn
python setup.py build_ext --inplace
cd ..

