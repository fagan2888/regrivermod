cd regrivermod
rm *.c
rm *.so
rm *.swp
rm *.html
rm -rf build
#zip -r regrivermod.zip ~/Dropbox/Model/regrivermod
python setup.py build_ext --inplace
cd ..
 
cd econlearn
rm *.c
rm *.so
rm *.swp
rm *.html
rm -rf build
#zip -r econlearn.zip ~/Dropbox/Model/econlearn
python setup.py build_ext --inplace
cd ..

#python multyvac_setup.py
