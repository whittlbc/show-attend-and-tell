import re
import h5py
import json
import numpy as np
import tensorflow as tf
from collections import Counter
from core.definitions import *
from core.utils import save_pickle
from scipy import ndimage
from core.vggnet import Vgg19

batch_size = 100  # get from somewhere more sharable
max_caption_words = 15
caption_vec_len = max_caption_words + 2  # +2 to account for <START> and <END>


def normalize_image(arr):
  return (1.0 * arr) / 255


def format_caption(caption):
  cap = re.sub('[.,"\')(]', '', caption).replace('&', 'and').replace('-', ' ')
  split_cap = cap.split()

  if len(split_cap) > max_caption_words:
    return None

  return ' '.join([w.strip() for w in cap.split()]).lower()


def format_split(split):
  # Get the raw downloaded caption data for the provided split
  with open('{}/captions_{}2014.json'.format(annotations_dir, split)) as f:
    caption_data = json.load(f)

  image_id_to_filename = {img['id']: img['file_name'] for img in caption_data['images']}

  # Get caption annotations sorted by image_id
  annotations = caption_data['annotations']
  annotations.sort(key=lambda a: a['image_id'])

  data = []
  for a in annotations:
    # Format caption by stripping/subbing out unwanted chars
    caption = format_caption(a['caption'])

    # If captions is None, it means it was too long, so skip the annotation.
    if not caption:
      continue

    image_id = a['image_id']
    image_path = '{}/{}2014_resized/{}'.format(image_dir, split, image_id_to_filename[image_id])

    data.append({
      'caption': caption,
      'image_id': image_id,
      'image_path': image_path
    })

  return data


def vectorize_cap(caption, word_to_index):
  vec = [word_to_index['<START>']]

  for w in caption.split(' '):
    if w in word_to_index:
      vec.append(word_to_index[w])

  vec.append(word_to_index['<END>'])

  # Pad vector with nulls to desired length
  for j in range(caption_vec_len - len(vec)):
    vec.append(word_to_index['<NULL>'])

  return one_hot(vec, len(word_to_index))


def one_hot(vector, vocab_len):
  arr = []

  for num in vector:
    row = [0] * vocab_len
    row[num] = 1
    arr.append(row)

  return arr


def build_vocab(data, word_count_threshold=1):
  captions = [d['caption'] for d in data]
  counter = Counter()

  # Populate a counter to measure word occurrences across all captions
  for caption in captions:
    for word in caption.split(' '):
      counter[word] += 1

  # We only want the words that occur more than our specified threshold
  words = [w for w in counter if counter[w] >= word_count_threshold]

  # Go ahead and populate vocab map with special words
  vocab = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2
  }

  # Populate the rest of the vocab map (this is our word_to_index map)
  i = 3
  for word in words:
    vocab[word] = i
    i += 1

  return vocab


def get_split_data():
  train_data = format_split('train')
  val_test_data = format_split('val')

  len_val_test_data = len(val_test_data)

  val_cutoff = int(0.1 * len_val_test_data)
  test_cutoff = int(0.2 * len_val_test_data)

  val_data = val_test_data[:val_cutoff]
  test_data = val_test_data[val_cutoff:test_cutoff]

  data = {
    'train': train_data,
    'val': val_data,
    'test': test_data
  }

  return data


def create_split_dataset(split, annotations, f, word_to_index, vggnet, sess):
  g = f.create_group(split)

  num_captions = len(annotations)
  num_images = len({a['image_id']: None for a in annotations})

  captions = g.create_dataset('captions',
                              shape=(num_captions, caption_vec_len, len(word_to_index)),
                              dtype=np.float32)

  images = g.create_dataset('images',
                            shape=(num_images, image_height, image_width, len(image_color_repr)),
                            dtype=np.float32)

  image_idxs = g.create_dataset('image_idxs', shape=(num_captions,), dtype=np.int32)

  features = g.create_dataset('features', shape=(num_images, 196, 512), dtype=np.float32)

  image_id_to_idx = {}

  for i, data in enumerate(annotations):
    image_id = data['image_id']
    image_idx = image_id_to_idx.get(image_id)

    if image_idx is None:
      image_idx = len(image_id_to_idx)
      image_id_to_idx[image_id] = image_idx

      images[image_idx] = normalize_image(ndimage.imread(data['image_path'], mode=image_color_repr))

      if image_idx % batch_size and image_idx > 0:
        end_idx = image_idx
        start_idx = end_idx - batch_size

        image_batch = images[start_idx:end_idx]

        feats = sess.run(vggnet.features, feed_dict={vggnet.images: image_batch})

        # TODO: think this assignment might break....check this
        features[start_idx:end_idx, :] = feats

    image_idxs[i] = image_idx
    captions[i] = vectorize_cap(data['caption'], word_to_index)

  # TODO: Delete the 'images' and 'image_idxs' datasets now that we don't need them anymore?


if __name__ == '__main__':
  # Get train, val, and test data
  split_data = get_split_data()

  # Create a word_to_index map based on the vocab in the train data ONLY
  word_to_index = build_vocab(split_data['train'])

  # Save the word_to_index map (will need later for CaptionGenerator)
  save_pickle(word_to_index, word_to_index_path)

  # Our hdf5 dataset that will hold all of ze data
  dataset = h5py.File(dataset_path, 'w')

  # Extract conv5_3 feature vectors
  vggnet = Vgg19(vgg_model_path)
  vggnet.build()

  # Init a new tensorflow session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # For each split, create an hdf5 group with nested datasets: captions, images, image_idxs
  for split, annotations in split_data.iteritems():
    create_split_dataset(split, annotations, dataset, word_to_index, vggnet, sess)

  # We done here.
  dataset.close()