from os import listdir
from os.path import isfile, join

path_images = './images'
path_structure = './structure'

images = sorted([f for f in listdir(path_images) if isfile(join(path_images, f))])
structure = sorted([f for f in listdir(path_structure) if isfile(join(path_structure, f))])

images = [int(img.strip('Places365_val').strip('.jpg')[3:]) for img in images]
structure = [int(struct.strip('Places365_val').strip('.jpg')) for struct in structure]

def find_missing(lst): 
    return sorted(set(range(lst[0], lst[-1])) - set(lst))
    
print(find_missing(images))
print(find_missing(structure))