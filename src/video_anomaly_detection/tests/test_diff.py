
import os.path
import pkg_resources
import pytest
import tempfile

import hickle
import numpy as np
import pandas as pd
import skvideo.io
import prednet.tests
import prednet.data_input
import prednet.evaluate
import prednet.train

import video_anomaly_detection.diff
import video_anomaly_detection.cli


def test_black(capsys):
  array = np.zeros((32, 8, 8, 3))
  array[-1, :, :, :] = 1
  with tempfile.TemporaryDirectory() as tempdirpath:
    prednet.data_input.save_array_as_hickle(array, ['black' for frame in array], tempdirpath)
    for filename in ('X_train.hkl', 'X_validate.hkl', 'X_test.hkl',
                     'sources_train.hkl', 'sources_validate.hkl', 'sources_test.hkl'):
      assert os.path.exists(os.path.join(tempdirpath, filename))
    for split in ('train', 'validate', 'test'):
      assert (hickle.load(os.path.join(tempdirpath, 'X_{}.hkl'.format(split))).shape[0] ==
              len(hickle.load(os.path.join(tempdirpath, 'sources_{}.hkl'.format(split)))))
    with capsys.disabled():
      prednet.train.train_on_hickles(tempdirpath,
                                     number_of_epochs=4, steps_per_epoch=8,
                                     path_to_save_weights_hdf5=os.path.join(tempdirpath, 'zero_weights.hdf5'),
                                     path_to_save_model_json=os.path.join(tempdirpath, 'prednet_model.json'))
      weights_path = os.path.join(tempdirpath, 'zero_weights.hdf5')
      assert os.path.exists(weights_path)
      prednet.evaluate.evaluate_on_hickles(tempdirpath,
                                           path_to_save_prediction_scores='prediction_scores.txt',
                                           path_to_model_json=os.path.join(tempdirpath, 'prednet_model.json'),
                                           weights_path=weights_path,
                                           RESULTS_SAVE_DIR=tempdirpath)
      save_path = '/tmp/videoAnomalies'
      video_anomaly_detection.diff.mse_test(tempdirpath, os.path.join(tempdirpath, 'prednet_model.json'),
                                            weights_path,
                                            save_path=save_path)
      test_results = pd.read_pickle(os.path.join(save_path, 'test_results.pkl.gz'))
      model_mse = test_results['model_mse']
      assert type(model_mse) is pd.Series
      if model_mse.shape != (8,):
        raise Exception(model_mse.shape, model_mse)
      assert np.count_nonzero(model_mse.iloc[:-1]) == 0
      assert model_mse.iloc[-1] > 0
    assert os.path.exists(os.path.join(tempdirpath, 'prednet_model.json'))


@pytest.mark.skipif(not skvideo.io.io._HAS_FFMPEG, reason='We cannot test loading a video without the video-loading library installed.')
def test_single_video():
  videoFile = pkg_resources.resource_filename('prednet.tests', os.path.join('resources', 'black.mpg'))
  assert os.path.exists(videoFile)
  video_anomaly_detection.cli.main([videoFile, '--number-of-epochs', '4', '--steps-per-epoch', '8'])


class StubCapSys:
  def disabled(self):
    import contextlib
    return contextlib.suppress(*[])


if __name__ == "__main__":
  """
  If having GPU problems, try running with CUDA_VISIBLE_DEVICES= to run on CPU.
  """
  test_single_video()
  test_black(StubCapSys())
