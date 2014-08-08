import multyvac as mv
import os

folder = '/home/nealbob/Dropbox/Model/regrivermod/'
filelist = os.listdir(folder)

river = mv.layer.get('river')
river.rm('/regrivermod')

for f in filelist:
    river.put_file(folder + f , '/regrivermod/' + f)
