import time
import platform

import cv2
from PIL import Image
import numpy as np
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import pylab
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import scipy
import tensorflow as tf
from tqdm import tqdm
# from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
import glob
import os
# os.environ['TF_KERAS'] = '1'
import csv
# import efficientnet.tfkeras
from keras_radam import RAdam
# from FractionalPooling2D import FractionalPooling2D
from sklearn.utils import class_weight
import efficientnet.tfkeras as efn
from tensorflow.python.keras.models import Sequential

def my_InceptionResNetV2(input_shape=(299,299,3),classes=5):
    model = Sequential()
    model.add(InceptionResNetV2(include_top=False, weights=None, input_tensor=None,
                              input_shape=input_shape, pooling='avg'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='elu'))
    model.add(Dense(1, activation='linear'))
    model.name = 'myInceptionResNetV2'
    return model

def tta(image,model):
    datagen=ImageDataGenerator()
    all_images=np.expand_dims(image,0)
    hori_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"flip_horizontal":True}),axis=0)
    vert_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"flip_vertical":True}),axis=0)
    rotated_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"theta":15}),axis=0)
    all_images=np.append(all_images,hori_image,axis=0)
    all_images=np.append(all_images,vert_image,axis=0)
    all_images=np.append(all_images,rotated_image,axis=0)
    prediction=model.predict(all_images)
    # print(prediction)
    return np.mean(prediction)


class kappa_call(Callback):

    def __init__(self, filepath, validation_data=None, max_nums=2):
        super(kappa_call, self).__init__()
        self.filepath = filepath
        self.x_val, self.y_val = validation_data
        self.best = -np.Inf
        self.save_models = []
        self.max_num = max_nums
        # self.df = pd.DataFrame()

    def on_epoch_end(self, epoch, logs=None):
        # valid_matrix = np.zeros((5, 5), dtype=np.int)
        # y_val = np.argmax(self.y_val, axis=1)
        y_val = self.y_val
        y_pred = np.empty((len(y_val),), dtype=np.int)
        for i, x in enumerate(self.x_val):
            x = x.reshape(( x.shape[0], x.shape[1], x.shape[2]))
            # out = tta_prediction(self.model,x,2)
            out=tta(x,self.model)
            if out < 0.5:
                y_pred[i] = 0
            elif out < 1.5:
                y_pred[i] = 1
            elif out < 2.5:
                y_pred[i] = 2
            elif out < 3.5:
                y_pred[i] = 3
            else:
                y_pred[i] = 4
            # y_pred[i] = np.argmax(out)
            # valid_matrix[y_val[i], np.argmax(out)] += 1
        cm = confusion_matrix(y_val, y_pred)
        val_kappa = cohen_kappa_score(y_val, y_pred, weights='quadratic')
        print(cm)
        print(val_kappa)

        # self.df = self.df.append(pd.Series(['epoch', int(epoch + 1)]), ignore_index=True)
        # for row in valid_matrix:
        #     self.df = self.df.append(pd.Series(row), ignore_index=True)
        # self.df = self.df.append(pd.Series(['val_kappa', val_kappa]), ignore_index=True)
        # self.df = self.df.append(pd.Series(['val_acc', valid_matrix.trace() / valid_matrix.sum()]), ignore_index=True)
        # self.df = self.df.append(pd.Series([' ']), ignore_index=True)
        # csv_path = os.path.join(os.path.split(self.filepath)[0], self.model.name + '.csv')
        # self.df.to_csv(csv_path, index=False, header=None)

        if val_kappa > self.best:
            print('\nEpoch {:05d}: kappa improved from {:0.5f} to {:0.5f},'
                  .format(epoch + 1, self.best, val_kappa, ))
            self.best = val_kappa
            filepath = self.filepath.format(epoch=epoch + 1, kappa=self.best)
            if len(self.save_models) < self.max_num:
                self.save_models.append(filepath)
            else:
                os.remove(self.save_models.pop(0))
                self.save_models.append(filepath)
            self.model.save(filepath)
            print('\nsaving model to {}'.format(filepath))
        else:
            print('\nEpoch {:05d}: kappa did not improve from {:0.5f}'
                  .format(epoch + 1, self.best))

IMG_SIZE = 300
SEED = 200
BATCH_SIZE = 8




def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def circle_crop_v2(img):
    # img = cv2.imread(img)
    # img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    # img = crop_image_from_gray(img)

    return img


def ben_color2(image_path, sigmaX=10, scale=270):
    # image = cv2.imread(image_path)

    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
    # gray_img = clahe.apply(gray_img)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    bgr = cv2.imread(image_path)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    x = image[image.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() // 2
    s = scale * 1.0 / r
    image = cv2.resize(image, (0, 0), fx=s, fy=s)
    # image = crop_image_from_gray(image)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX),
                            -4, 128)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = circle_crop_v2(image)

    # image=normalize(image)
    # image = cv2.fastNlMeansDenoisingColored(image,None,20,10,7,21)

    return image


def preprocess_image(image_path, desired_size=300):
    image = ben_color2(image_path, sigmaX=10)
    # image =image.reshape((1,IMG_SIZE,IMG_SIZE,3))
    return image


def load_csv(filename):

    # read from csv file
    images, labels = [], []
    with open(filename) as f:
        reader = csv.reader(f)
        result = list(reader)
        for row in result[1:]:
            # img=os.path.join('/media/td/B4DAA25FDAA21E1C/isbi',row[2])
            img = '/media/td/B4DAA25FDAA21E1C/isbi' + row[2].replace('\\','/')
            if row[4] != '':
                label = row[4]
            if row[5] != '':
                label = row[5]
            label = int(label)

            images.append(img)
            labels.append(label)
    return images, labels



def build_model2():
    effnet = efn.EfficientNetB5(weights=None,
                                include_top=False,
                                input_shape=(IMG_SIZE, IMG_SIZE, 3))
    effnet.load_weights(
        '/home/td/桌面/efficientnet-b5_imagenet_1000_notop.h5')
    for i, layer in enumerate(effnet.layers):
        if "batch_normalization" in layer.name:
            effnet.layers[i] = layers.BatchNormalization(groups=32,
                                                         axis=-1,
                                                         epsilon=0.00001)
    model = Sequential()
    model.add(effnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1024, activation="elu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="elu"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['acc'])
    return model


def get_preds(model, x_val):
    preds = []
    for x in x_val:
        x = x.reshape((1, IMG_SIZE, IMG_SIZE, 3))
        x = tf.cast(x, tf.float32)
    preds=model.predict(x_val)
    # return np.concatenate(preds).ravel()
    return preds

def get_labels(model, y_val):
    labels = []
    for y in y_val:
        labels.append(y)
    return labels




def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  # set range for random zoom
        rotation_range=360,
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

def myDataload_kaggle(data_root=r'/home/td/Diabetic_Retinopat_Detection/data', csv_path=None, select=None, re_size=(300, 300)):
    select_nums = [1500, 1000, 1500, 800, 700] if select is None else select
    # csv_path_u = '/'.join(csv_path.split('\\')) if platform.system() is not 'Windows' else csv_path
    csv_path_u='/home/td/Diabetic_Retinopat_Detection/data/trainLabels.csv'
    csv_file = pd.read_csv(csv_path_u)
    df = pd.DataFrame(csv_file)
    dfs = pd.DataFrame()
    for i in range(5):
        dft = df.loc[df['level'] == i]
        dfs = dfs.append(dft.sample(select_nums[i]))
    df = dfs
    true_labels = df['level'].values.tolist()
    samples = len(true_labels)
    images = np.empty((samples, re_size[0], re_size[1], 3), dtype=np.uint8)
    lables = np.empty(samples, dtype=np.float)

    for i, p in enumerate(tqdm(df['image'])):
        im_path = os.path.join(data_root, 'train', p + '.jpeg')
        im_path = '/'.join(im_path.split('\\')) if platform.system() is not 'Windows' else im_path
        img = preprocess_image(im_path)
        img = img.reshape(1, re_size[0], re_size[1], 3)
        images[i, :, :, :] = img
        lables[i] = int(true_labels[i])

    return images, lables


def weibiaoqian_data(file_path,data_root=r'/home/td/PycharmProjects/isbi/data/test'):
    images, labels = [], []
    with open(file_path) as f:
        reader = csv.reader(f)
        result = list(reader)
        for row in result:
            label=row[1]
            f_name=row[0]+'.jpg'
            dir_name=f_name[:3]
            dir_path=os.path.join(data_root,dir_name)
            img=os.path.join(dir_path,f_name)
            images.append(img)
            labels.append(label)
    return images,labels

#添加伪标签
wei_images,wei_labels=weibiaoqian_data("weibiaoqian.csv")

N=len(wei_images)
wei_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, x in enumerate(tqdm(wei_images)):
    wei_train[i, :, :, :] = preprocess_image(x)


train_df = pd.read_csv('/home/td/桌面/aptos2019-blindness-detection/train.csv')
y_train1 = train_df['diagnosis'].values
N = train_df.shape[0]
x_train1 = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    x_train1[i, :, :, :] = preprocess_image(
        f'/home/td/桌面/aptos2019-blindness-detection/train_images/{image_id}.png'
    )


x_tra, y_train2 = load_csv(
    '/media/td/B4DAA25FDAA21E1C/isbi/regular-fundus-training/regular-fundus-training.csv'
)
x_val, y_validation = load_csv(
    '/media/td/B4DAA25FDAA21E1C/isbi/regular-fundus-validation/regular-fundus-validation.csv'
)

N = len(x_tra)
x_train2 = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, x in enumerate(tqdm(x_tra)):
    x_train2[i, :, :, :] = preprocess_image(x)

M = len(x_val)
x_validation = np.empty((M, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, x in enumerate(tqdm(x_val)):
    x_validation[i, :, :, :] = preprocess_image(x)

y_train = np.concatenate((y_train1,y_train2,wei_labels))
x_train = np.concatenate((x_train1, x_train2,wei_train))

y_validation = np.array(y_validation)
y_train = np.array(y_train)

wts = class_weight.compute_class_weight('balanced', np.unique(y_train),
                                        y_train)

model = build_model2()

model.compile(loss='mse', optimizer=Adam(lr=0.0005), metrics=['acc'])

data_generator = create_datagen().flow(x_train,
                                       y_train,
                                       batch_size=BATCH_SIZE,
                                       seed=77)


st = time.strftime('%Y%m%d_%H%M%S', time.localtime())
model_dir = os.path.join('.', 'csymodels', st)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, model.name + '_{epoch:03d}_kappa-{kappa:.5f}.h5')

kappa_callback = kappa_call(model_path, validation_data=(x_validation, y_validation))
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=15)
lr = ReduceLROnPlateau(monitor='val_loss',
                        factor=0.5,
                        patience=6,
                        verbose=1,
                        mode='auto',
                        min_delta=0.0001)

callbacks_list = [kappa_callback,lr]
history2 = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
    epochs=100,
    validation_data=(x_validation, y_validation),
    class_weight=wts,
    validation_steps=x_validation.shape[0] // BATCH_SIZE,
    callbacks=callbacks_list,
)