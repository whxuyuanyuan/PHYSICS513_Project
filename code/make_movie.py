import os
os.system(
    'ffmpeg -framerate 10 -start_number 0 -i ../output/test/images/image%04d.jpg -qscale 1 -vf scale=1500:1000 ../output/test/movie.avi >/dev/null 2>&1')
