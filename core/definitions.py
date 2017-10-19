import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = basedir + '/data'
dataset_path = data_dir + '/dataset.hdf5'
vgg_model_path = data_dir + '/imagenet-vgg-verydeep-19.mat'
annotations_dir = data_dir + '/annotations'
image_dir = basedir + '/image'
image_height = 224
image_width = 224
image_color_repr = 'RGB'