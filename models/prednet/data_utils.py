import numpy as np
import keras
import hickle as hkl
from keras import backend as K


class SequenceGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_data_file, source_file, nt, y_data_file=None,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.X = hkl.load(X_data_file)
        if y_data_file:
            self.y = hkl.load(y_data_file)
        else:
            self.y = None
        # source for each image so when creating sequences can assure
        # that consecutive frames are from same video.
        self.sources = hkl.load(source_file)
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, \
            'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction', 'classification'}, \
            'output_mode must be in {error, prediction, classification}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':
            # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array(
                [i for i in range(self.X.shape[0] - self.nt)
                    if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':
            # create sequences where each unique frame is in
            # at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                curr_source = self.sources[curr_location]
                curr_source_p_nt = self.sources[curr_location + self.nt - 1]
                # if the next nt frames are from the same file, the curr_location can be used
                # start of a sequence of nt length. Else, we increment by one, until we find 
                # the next such position
                if curr_source == curr_source_p_nt:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:
            # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        print("found %d possible starts in %s" % (len(self.possible_starts), X_data_file))

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_x = np.zeros(
            (self.batch_size, self.nt) + self.im_shape,
            np.float32)
        index_array = np.random.choice(self.possible_starts, self.batch_size)
        for i, idx in enumerate(index_array):
            # idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])

        if self.output_mode == 'error':
            # model outputs errors, so y should be zeros
            batch_y = np.zeros(self.batch_size, np.float32)
        elif self.output_mode == 'prediction':
            # output actual pixels
            batch_y = batch_x
        elif self.output_mode == 'classification':
            batch_y = self[idx:idx+self.nt]

        return batch_x, batch_y

    def preprocess(self, X):
        """ Currently this just scaled the images to [0,1]

        Arguments:
            X {np.array} -- array of images n_images * height * width * depth

        Returns:
            [type] -- [description]
        """

        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros(
            (self.N_sequences, self.nt) + self.im_shape,
            np.float32)
        y_all = np.zeros(
            (self.N_sequences, self.nt), np.float32
        )
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
            y_all[i] = self.y[idx:idx+self.nt]
        return X_all, y_all


class TestsetGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_file, source_file, nt, y_data_file=None,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.X = hkl.load(data_file)
        if y_data_file:
            self.y = hkl.load(y_data_file)
        else:
            self.y = None
        # source for each image so when creating sequences can assure
        # that consecutive frames are from same video
        self.sources = hkl.load(source_file)
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert output_mode in {'error', 'prediction', 'classification'}, \
            'output_mode must be in {error, prediction, classification}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        self.possible_starts = [0]
        current_source = self.sources[0]
        for i, source in enumerate(self.sources):
            if source != current_source:
                self.possible_starts.append(i)
                current_source = source

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:
            # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_x = np.zeros(
            (self.batch_size, self.nt) + self.im_shape,
            np.float32)
        index_array = np.random.choice(self.possible_starts, self.batch_size)
        for i, idx in enumerate(index_array):
            # idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':
            # model outputs errors, so y should be zeros
            batch_y = np.zeros(self.batch_size, np.float32)
        elif self.output_mode == 'prediction':
            # output actual pixels
            batch_y = batch_x
        elif self.output_mode == 'classification':
            batch_y = self[idx:idx+self.nt]

        return batch_x, batch_y

    def preprocess(self, X):
        """ Currently this just scaled the images to [0,1]

        Arguments:
            X {np.array} -- array of images n_images * height * width * depth

        Returns:
            [type] -- [description]
        """

        return X.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros(
            (self.N_sequences, self.nt) + self.im_shape, np.float32)
        y_all = np.zeros(
            (self.N_sequences, self.nt), np.float32
        )
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
            y_all[i] = self.y[idx:idx+self.nt]
        return X_all, y_all
