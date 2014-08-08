#picloud env rsync ~/Dropbox/Model/chapter5/picloud/ river2:/home/picloud
#picloud env ssh river2 rm *.c 
#picloud env ssh river2 python setup.py build_ext --inplace
#picloud env save river2
#picloud exec -e river2 -c 4 -t f2 python master.py 0 1
picloud exec -e river2 -c 4 -t f2 python master.py 34 35
picloud exec -e river2 -c 4 -t f2 python master.py 45 46
