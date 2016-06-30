from __future__ import print_function

try:
    import gpuselect
except:
    print('gpuselect is not installed')

from PIL import Image
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.models import Sequential

from data import load_train_data, load_test_data

img_rows = 64
img_cols = 80
n_channel = 1
target_size = (img_cols, img_rows)

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((n_channel, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    conv10 = Flatten()(conv10)
    conv10 = Dense(1, activation='sigmoid')(conv10)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_keras_example_net():
    # inputs = Input((n_channel, img_rows, img_cols))
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(n_channel, img_cols, img_rows)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Convolution2D(64, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        img = Image.fromarray(imgs[i, 0])
        img = img.resize(target_size)
        imgs_p[i, 0] = img_to_array(img, dim_ordering='th')
    return imgs_p


def k_fold_train(imgs_train, imgs_mask_train, n_fold=5):
    from sklearn.cross_validation import KFold
    kf = KFold(imgs_train.shape[0], n_folds=n_fold)
    current = 0
    for train_index, test_index in kf:
        # model = get_unet()
        model = get_keras_example_net()
        a_train_imgs = np.take(imgs_train, train_index, axis=0)
        a_train_mask = np.take(imgs_mask_train, train_index, axis=0)
        a_valid_imgs = np.take(imgs_train, test_index, axis=0)
        a_valid_mask = np.take(imgs_mask_train, test_index, axis=0)
        model_checkpoint = ModelCheckpoint('unet_fold%s.hdf5' % current, monitor='val_loss', save_best_only=True)
        model.fit(a_train_imgs, a_train_mask, validation_data=(a_valid_imgs, a_valid_mask),
                  batch_size=32, nb_epoch=5, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])
        current += 1


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    y_hat_train = np.max(imgs_mask_train, axis=(1, 2, 3))
    print(y_hat_train.shape, np.unique(y_hat_train), np.sum(y_hat_train))
    y_hat_train_sums = np.sum(imgs_mask_train, axis=(1, 2, 3))
    y_hat_train_sums = np.nonzero(y_hat_train_sums)[0]
    print(y_hat_train_sums.shape, np.min(y_hat_train_sums), np.max(y_hat_train_sums), np.mean(y_hat_train_sums))
    y = np.bincount(y_hat_train_sums)
    ii = np.nonzero(y)[0]
    count = y[ii]
    from matplotlib import pyplot as plt
    plt.plot(ii, count)
    plt.show()
    raw_input()

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    n_fold = 5
    k_fold_train(imgs_train, y_hat_train, n_fold=n_fold)

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    # model = get_unet()
    model = get_keras_example_net()
    results = []
    for i in range(n_fold):
        model.load_weights('unet_fold%s.hdf5' % i)
        print('-' * 30)
        print('%s Predicting masks on test data...' % i)
        print('-' * 30)
        imgs_mask_test = model.predict(imgs_test, verbose=1)
        results.append(imgs_mask_test)
    imgs_mask_test = reduce(lambda x, y: x + y, results) / n_fold
    np.save('imgs_mask_test_nfold.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
