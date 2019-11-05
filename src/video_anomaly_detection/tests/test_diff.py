
import hickle
import os.path
import tempfile
import numpy as np

import prednet.data_input
import prednet.evaluate
import prednet.train


def test_black(capsys):
  array = np.zeros((32, 8, 8, 3))
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.save_array_as_hickle(array, ['black' for frame in array], tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    for split in ('train', 'validate', 'test'):
      assert hickle.load(os.path.join(tempdirpath, 'X_{}.hkl'.format(split))).shape[0] == len(hickle.load(os.path.join(tempdirpath, 'sources_{}.hkl'.format(split))))
    with capsys.disabled():
      prednet.train.train_on_hickles(tempdirpath, tempdirpath, array.shape[1], array.shape[2],
                                     number_of_epochs=4, steps_per_epoch=8,
                                     weights_file='zero_weights.hdf5')
      weights_path = os.path.join(tempdirpath, 'zero_weights.hdf5')
      assert os.path.exists(weights_path)
      prednet.evaluate.evaluate_json_model(tempdirpath, tempdirpath, tempdirpath,
                                           path_to_save_prediction_scores='prediction_scores.txt',
                                           weights_path=weights_path)
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))
