
import os
import urllib

import time


def main():
    errors = []
    with open("revision-corpus/vis10cat.txt") as f:
        i = 0
        for line in f:
            [chart_type, url] = line.split('\t')
            url = url.rstrip('\r\n')
            # print chart_type
            # print url

            dir_path = "Charts/" + chart_type
            if not os.path.isdir(dir_path):
                # Create the Directory
                print('Directory doesn\'t exist - Creating ' + dir_path)
                os.makedirs(dir_path)
                i = 0

            # Download url and put it inside the directory
            i = i + 1
            file_type = url[-3:]
            file_name = 'Charts/' + chart_type + '/' + str(i) + '_' + chart_type + '.' + file_type
            print(' Downloading file : ' + file_name)
            time.sleep(0.1)
            try:
                urllib.urlretrieve (url, file_name)
            except IOError as e:
                print("Not downloading " + file_name);
                errors.append(file_name)
            except Exception as e:
                print("Not downloading " + file_name);
                errors.append(file_name)
            # wget.download(url, file_name)
    with open('errors.txt', 'wa') as error_file:
        for file in errors:
            error_file.write(file_name + "\t" + url + "\n")

if __name__ == '__main__':
    main()
