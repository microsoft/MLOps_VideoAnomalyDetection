'''
Run trained PredNet on UCSD sequences to create data for anomaly detection
'''

import argparse
import os
import shutil

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prednet.evaluate
import skvideo.io
import tensorflow as tf
# from keras import backend as K
from prednet.data_utils import TestsetGenerator
from scipy.ndimage import gaussian_filter

matplotlib.use('Agg')


def show_anomalies_as_overlay_single_video(path_to_video,
                                           number_of_epochs=150, steps_per_epoch=125,
                                           ):
  path_to_save_overlay_video = os.path.splitext(path_to_video)[0] + '.overlay.' + os.path.splitext(path_to_video)[1]
  predictedFrames = prednet.evaluate.get_predicted_frames_for_single_video(path_to_video, number_of_epochs, steps_per_epoch)
  actualFrames = skvideo.io.vread(path_to_video)
  if actualFrames.shape != predictedFrames.shape:
    raise Exception(actualFrames.shape, predictedFrames.shape)
  overlayVideo = np.empty(actualFrames.shape)
  for index,frame in enumerate(overlayVideo):
    overlayVideo[index, :] = overlay_frame_error(predictedFrames[index], frame)
  skvideo.io.vwrite(path_to_save_overlay_video, overlayVideo)


def mse_test(DATA_DIR, model_json_path, weights_hdf5_path, lengthOfVideoSequences=8, save_path='.'):
  # Define args
  parser = argparse.ArgumentParser(description='Process input arguments')
  parser.add_argument('--out_data', default='./data/video/', type=str, dest='out_data',
                      help='path to data and annotations (annotations should be in <data_dir>/<dataset>/Test/<dataset>.m')
  parser.add_argument('--n_plot', default=0, type=int, dest='n_plot', help='How many sample sequences to plot')
  parser.add_argument('--batch_size', default=10, type=int, dest='batch_size', help='How many epochs per batch')
  parser.add_argument('--N_seq', default=None, type=int, dest='N_seq', help='how many videos per epoch')
  parser.add_argument('--save_prediction_error_video_frames', action='store_true', dest='save_prediction_error_video_frames',
                      help='how many videos per epoch')

  # args = parser.parse_args()
  nt = lengthOfVideoSequences
  batch_size = 4
  N_seq = None

  n_plot = 1
  save_prediction_error_video_frames = True

  if tf.test.is_gpu_available():
      print("We have a GPU")
  else:
      print("Did not find GPU")

  # check/create path for saving output
  # extent data_dir for current dataset
  # data_dir = os.path.join(data_dir, dataset, 'Test')
  # os.makedirs(data_dir, exist_ok=True)

  # load the dataset
  test_file = os.path.join(DATA_DIR, 'X_test.hkl')
  test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

  test_model, data_format = prednet.evaluate.make_evaluation_model(model_json_path, weights_hdf5_path,
                                                                   nt=lengthOfVideoSequences)

  # Define Generator for test sequences
  test_generator = TestsetGenerator(test_file, test_sources, nt, data_format=data_format, N_seq=N_seq)
  X_test = test_generator.create_all()

  # Apply model to the test sequences
  X_hat = test_model.predict(X_test, batch_size)
  if data_format == 'channels_first':
      X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
      X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

  # Calculate MSE of PredNet predictions vs. using last frame, and aggregate across all frames in dataset
  model_mse = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
  prev_mse = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )  # this simply using the last frame

  os.makedirs(save_path, exist_ok=True)

  #  Write results to prediction_scores.txt
  f = open(os.path.join(save_path, 'prediction_scores.txt'), 'w')
  f.write("Model MSE: %f\n" % model_mse)
  f.write("Previous Frame MSE: %f" % prev_mse)
  f.close()

  # Compare MSE of PredNet predictions vs. using last frame, without aggregating across frames
  model_err = X_test - X_hat
  model_err[:, 0, :, :, :] = 0  # first frame doesn't count
  model_mse = np.mean( (model_err)**2, axis=(2,3,4) )  # look at all timesteps except the first
  model_p_50 = np.percentile((model_err)**2, 50, axis=(2,3,4))
  model_p_75 = np.percentile((model_err)**2, 75, axis=(2,3,4))
  model_p_90 = np.percentile((model_err)**2, 90, axis=(2,3,4))
  model_p_95 = np.percentile((model_err)**2, 95, axis=(2,3,4))
  model_p_99 = np.percentile((model_err)**2, 99, axis=(2,3,4))
  model_std = np.std((model_err)**2, axis=(2,3,4))

  # now we flatten them so that they are all in one column later
  model_mse = np.reshape(model_mse, np.prod(model_mse.shape))
  model_p_50 = np.reshape(model_p_50, np.prod(model_mse.shape))
  model_p_75 = np.reshape(model_p_75, np.prod(model_mse.shape))
  model_p_90 = np.reshape(model_p_90, np.prod(model_mse.shape))
  model_p_95 = np.reshape(model_p_95, np.prod(model_mse.shape))
  model_p_99 = np.reshape(model_p_99, np.prod(model_mse.shape))
  model_std = np.reshape(model_std, np.prod(model_mse.shape))

  prev_err = X_test[:, :-1] - X_test[:, 1:]  # simple comparison w/ prev frame as baseline for performance
  prev_err = np.insert(prev_err, 0, X_test[0,0].shape, axis=1)
  prev_mse = np.mean( (prev_err)**2, axis=(2,3,4) )  # look at all timesteps except the first
  prev_p_50 = np.percentile((prev_err)**2, 50, axis=(2,3,4))
  prev_p_75 = np.percentile((prev_err)**2, 75, axis=(2,3,4))
  prev_p_90 = np.percentile((prev_err)**2, 90, axis=(2,3,4))
  prev_p_95 = np.percentile((prev_err)**2, 95, axis=(2,3,4))
  prev_p_99 = np.percentile((prev_err)**2, 99, axis=(2,3,4))
  prev_std = np.std((prev_err)**2, axis=(2,3,4))

  # now we flatten them so that they are all in one column later
  prev_mse = np.reshape(prev_mse, np.prod(prev_mse.shape))
  prev_p_50 = np.reshape(prev_p_50, np.prod(prev_mse.shape))
  prev_p_75 = np.reshape(prev_p_75, np.prod(prev_mse.shape))
  prev_p_90 = np.reshape(prev_p_90, np.prod(prev_mse.shape))
  prev_p_95 = np.reshape(prev_p_95, np.prod(prev_mse.shape))
  prev_p_99 = np.reshape(prev_p_99, np.prod(prev_mse.shape))
  prev_std = np.reshape(prev_std, np.prod(prev_mse.shape))

  # save the results to a dataframe
  df = pd.DataFrame({'model_mse': model_mse,
                     'model_p_50': model_p_50, 'model_p_75': model_p_75, 'model_p_90': model_p_90, 'model_p_95': model_p_95,
                     'model_p_99': model_p_99,
                     'model_std': model_std,
                     'prev_mse': prev_mse,
                     'prev_p_50': prev_p_50, 'prev_p_75': prev_p_75, 'prev_p_90': prev_p_90, 'prev_p_95': prev_p_95, 'prev_p_99': prev_p_99,
                     'prev_std': prev_std})
  df.to_pickle(os.path.join(save_path, 'test_results.pkl.gz'))

  # Create plots for illustation of model performance
  if n_plot > 0:
      skip_frames_plot = 4
      print("Creating %s plots" % n_plot)
      # Plot some predictions
      # aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
      plt.figure(figsize=(nt//skip_frames_plot, 4*1.6))  # *aspect_ratio))
      gs = gridspec.GridSpec(4, nt//skip_frames_plot)
      gs.update(wspace=0., hspace=0.)

      plot_save_dir = os.path.join(save_path, 'prediction_plots')

      if os.path.exists(plot_save_dir):
          shutil.rmtree(plot_save_dir)
      os.makedirs(plot_save_dir)

  plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
  for i in plot_idx:
      for tt in range(nt):
          if tt % skip_frames_plot > 0:
              continue
          t = tt // skip_frames_plot

          err = np.abs(X_hat[i,tt] - X_test[i,tt])

          err_ov = gaussian_filter(err, 3)
          err_ov[err_ov < .1] = 0.0
          overlay = X_test[i,tt].copy()
          overlay[:,:,0] += err_ov[:,:,0]*5.0

          plt.subplot(gs[t])
          plt.imshow(X_test[i,tt], interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t == 0:
            plt.ylabel('Actual', fontsize=10)  # plot ylabel on left of first image

          plt.subplot(gs[t + nt//skip_frames_plot])
          plt.imshow(X_hat[i,tt], interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t == 0:
            plt.ylabel('Predicted', fontsize=10)

          plt.subplot(gs[t + nt//skip_frames_plot*2])
          plt.imshow(overlay, interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t == 0:
            plt.ylabel('Overlay', fontsize=10)

          # You can use this to also plot the previous video frame for comparison
          # plt.subplot(gs[t + nt*2])
          # plt.imshow(X_test[i,t - 1], interpolation='none')
          # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          # if t==0: plt.ylabel('Previous', fontsize=10)

          plt.subplot(gs[t + nt//skip_frames_plot*3])
          plt.imshow(err, interpolation='none')
          plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
          if t == 0:
            plt.ylabel('Abs. Error', fontsize=10)
          plt.xlabel(t, fontsize=6)

      plt.savefig(os.path.join(plot_save_dir,  'plot_' + str(i) + '.png'))
      plt.clf()

  # create frames that can be used for a video that shows anomalies as overlay
  if save_prediction_error_video_frames and n_plot > 0:
      movie_save_dir = os.path.join(save_path, 'PE_videoframes')

      if not os.path.exists(movie_save_dir):
          os.makedirs(movie_save_dir)

      for i in plot_idx:
          for tt in range(nt):
              overlay = overlay_frame_error(X_hat[i,tt], X_test[i,tt])

              plt.imshow(overlay, interpolation='none')
              plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                              labelbottom='off', labelleft='off')

              plt.savefig(os.path.join(movie_save_dir,  'frame_%02d_%03d.png' % (i, tt)))
              plt.close()


def overlay_frame_error(predictedFrame, actualFrame):
    assert predictedFrame.shape == actualFrame.shape
    err = np.abs(predictedFrame - actualFrame)

    err_ov = gaussian_filter(err, 3)
    err_ov[err_ov < .1] = 0.0
    overlay = actualFrame.copy()
    overlay[:,:,0] += err_ov[:,:,0]*5.0
    return overlay
