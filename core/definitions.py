import os

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_dir = basedir + '/data'
image_dir = basedir + '/image'
annotations_dir = data_dir + '/annotations'

dataset_path = data_dir + '/dataset.hdf5'
vgg_model_path = data_dir + '/imagenet-vgg-verydeep-19.mat'
word_to_index_path = data_dir + '/train/word_to_index.pkl'

image_height = 224
image_width = 224
image_color_repr = 'RGB'

feat_vec_len = 196
feat_vec_dim = 512
feat_batch_size = 20

max_caption_words = 15
caption_vec_len = max_caption_words + 2  # +2 to account for <START> and <END>