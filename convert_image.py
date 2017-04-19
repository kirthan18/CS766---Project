import os
from os.path import basename
from shutil import copyfile

from PIL import Image

directories = os.walk('Charts').next()[1]

for d in directories:
    src_directory = "Charts/" + d
    destination_directory = "jpg/" + src_directory

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for file in os.listdir("%s" % src_directory):
        try:
            file_name = basename(file)
            file_name = os.path.splitext(file_name)[0]
            src_file = os.path.join(src_directory, file)
            dest_file = os.path.join(destination_directory, file_name + ".jpg")

            if file.endswith(".png"):
                im = Image.open(src_file)
                im.convert('RGB').save(dest_file, 'JPEG')

                im = Image.open(dest_file)
                im.verify()
            elif file.endswith(".jpg"):
                print "copied - " + src_file

                im = Image.open(src_file)
                im.verify()

                copyfile(src_file, dest_file)
        except Exception as ex:
            print(ex)
