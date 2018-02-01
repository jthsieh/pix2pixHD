### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import numpy as np
from PIL import Image

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
#web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

root = '/mnt/hdd-persistent/cityscapes/synthetic'
os.makedirs(root, exist_ok=True)
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    path = data['path'][0]
    filename = os.path.basename(path)
    city = filename.split('_')[0]
    city_dir = os.path.join(root, opt.phase, city)
    os.makedirs(city_dir, exist_ok=True)
    for j in range(5):
      generated = model.inference(data['label'], data['inst'])
      syn = util.tensor2im(generated.data[0])
      output_filename = filename.replace('gtFine_labelIds', 'synthetic').split('.')[0] + '_{}.png'.format(j)
      output_path = os.path.join(city_dir, output_filename)
      img_pil = Image.fromarray(syn)
      img_pil.save(output_path)

    if (i+1) % 10 == 0:
      print('{} / {}'.format(i + 1, len(dataset)))
    #visualizer.save_images(webpage, visuals, img_path)

#webpage.save()
