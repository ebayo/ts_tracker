# Change classes of the database to include or delete classes
# If some images end up with no images, move them to "unused folder" for review


import argparse
import os


def read_class_file(class_file):
    ff = open(class_file, 'r')
    names = ff.readlines()
    ff.close()
    return [l[:-1] for l in names]


def idx_correspondence(old, new):
    ids = []
    for i in range(len(old)):
        if old[i] in new:
            ids.append(new.index(old[i]))
        else:
            ids.append(-1)
    return ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str,
                        help='path to the annotations directory')
    parser.add_argument('old_classes', type=str,
                        help='File with the name of the old classes, one per line, same used in database_split')
    parser.add_argument('new_classes', type=str,
                        help='File with the name of the new classes, one per line, same used in database_split')
    parser.add_argument('-n', '--new_folder', type=str,
                        help='Directory to save the new annotations, if not provides path/to/files_<num classes>')

    param = parser.parse_args()

    c_old = read_class_file(param.old_classes)
    c_new = read_class_file(param.new_classes)
    changes = idx_correspondence(c_old, c_new)
    # print(changes)

    if param.new_folder is None:
        if param.files.endswith('/'):
            new_folder = param.files[:-1] + '_' + str(len(c_new))
        else:
            new_folder = param.files + '_' + str(len(c_new))
    else:
        new_folder = param.new_folder

    os.makedirs(new_folder, exist_ok=True)

    for f in os.listdir(param.files):
        # print('Processing {}'.format(f))
        file = open(os.path.join(param.files, f), 'r')
        lines = file.readlines()
        file.close()
        file = open(os.path.join(new_folder, f), 'w')
        for l in lines:
            ll = str(changes[int(l[0])]) + l[1:]
            if ll[0] != '-':
                file.write(ll)
        file.close()
    print('done')

