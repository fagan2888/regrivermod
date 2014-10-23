#!/bin/bash
cd regrivermod
rm *.c
rm *.so
rm -rf build
python setup.py build_ext --inplace
cd ..
 
cd econlearn
rm *.c
rm *.so
rm -rf build
python setup.py build_ext --inplace
cd ..

#python multyvac_setup.py
