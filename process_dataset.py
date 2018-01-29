import glob
import os
import sys

root = '/mnt/hdd-persistent/cityscapes'
dest = os.path.join(root, 'data')

def copy_files(mode, data_type):
  data_dir = os.path.join(dest, '{}_{}'.format(mode, data_type))
  os.makedirs(data_dir, exist_ok=True)
  if data_type == 'img':
    folder = os.path.join(root, 'leftImg8bit', mode)
    for city_dir in glob.glob(os.path.join(folder, '*')):
      print(city_dir)
      os.system('cp {}/* {}'.format(city_dir, data_dir))
  elif data_type == 'inst':
    folder = os.path.join(root, 'gtFine', mode)
    for city_dir in glob.glob(os.path.join(folder, '*')):
      print(city_dir)
      os.system('cp {}/*_instanceIds.png {}'.format(city_dir, data_dir))
  elif data_type == 'label':
    folder = os.path.join(root, 'gtFine', mode)
    for city_dir in glob.glob(os.path.join(folder, '*')):
      print(city_dir)
      os.system('cp {}/*_labelIds.png {}'.format(city_dir, data_dir))

copy_files('test', 'inst')
