import h5py
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.definitions import dataset_path


def main():
  data = h5py.File(dataset_path, 'r')

  train_data = data.get('train')
  val_data = data.get('val')
  word_to_idx = data.get('word_to_idx')

  model = CaptionGenerator(word_to_idx, alpha_c=1.0)

  solver = CaptioningSolver(model, train_data, val_data,
                            n_epochs=20, batch_size=128, update_rule='adam',
                            learning_rate=0.001, print_every=1000, save_every=1,
                            model_path='model/lstm/', test_model='model/lstm/model-10',
                            print_bleu=True, log_path='log/')

  solver.train()


if __name__ == '__main__':
  main()