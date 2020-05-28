from random import randint

import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model
import numpy as np
import os
import cv2, csv
from math import ceil
from tqdm import tqdm
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import efficientnet.tfkeras as efn
from tensorflow.python.keras.optimizers import Adam
# def top_2_accuracy(in_gt, in_pred):
#     return top_k_categorical_accuracy(in_gt, in_pred, k=2)
from tensorflow.python.keras import backend as K
import efficientnet.tfkeras
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.python.keras.models import Sequential
# from tta_wrapper import tta_segmentation

IMG_SIZE = 300
NUM_CLASSES = 5
SEED = 77
TRAIN_NUM = -1
BATCH_SIZE = 4
print(1)

# model5=tf.keras.models.load_model('/home/td/PycharmProjects/isbi/models/20200318_114015/myInpResNetV2_029_kappa_tta-0.86160.h5', custom_objects=None, compile=False)
# model6=tf.keras.models.load_model('/home/td/PycharmProjects/isbi/models/20200318_114015/myInpResNetV2_097_kappa-0.85469.h5', custom_objects=None, compile=False)
#
# model1=load_model('/home/td/PycharmProjects/eyes/csymodels/20200316_145612/sequential_027_kappa-0.87212.h5')
# model2=load_model('/home/td/PycharmProjects/eyes/csymodels/20200316_145612/sequential_013_kappa-0.86358.h5')
# model3=load_model('/home/td/PycharmProjects/eyes/classify_models/20200317_232227/sequential_022_kappa-0.86550.h5')
# model4=load_model('/home/td/PycharmProjects/eyes/classify_models/20200317_232227/sequential_016_kappa-0.85637.h5')
model1=load_model('/home/td/PycharmProjects/eyes/5_foldmodels/20200319_000157/sequential_010_kappa-0.81749.h5')
model=load_model('/home/td/PycharmProjects/eyes/5_foldmodels/20200319_021001/sequential_1_015_kappa-0.83056.h5')
# tta_model = tta_segmentation(model, h_flip=True, rotation=(90, 270),
#                              h_shift=(-5, 5), merge='mean')
# model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['acc'])
# model.summary()
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

def tta_classify(image,model):
    datagen=ImageDataGenerator()
    all_images=np.expand_dims(image,0)
    hori_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"flip_horizontal":True}),axis=0)
    vert_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"flip_vertical":True}),axis=0)
    rotated_image=np.expand_dims(datagen.apply_transform(x=image,transform_parameters={"theta":15}),axis=0)
    all_images=np.append(all_images,hori_image,axis=0)
    all_images=np.append(all_images,vert_image,axis=0)
    all_images=np.append(all_images,rotated_image,axis=0)
    prediction=model.predict(all_images)
    prediction=np.sum(prediction,axis=0)
    # print(prediction)
    return np.argmax(prediction)

def tta_prediction(model, image, n_examples):
    """
        make a prediction for one image using test-time augmentation

        @param model: model for prediction
        @param image: input image to be predicted (nrows* ncols * nchns)
        @param n_exampls:

        @return yhat: predicted label for image (scalar int)
    """
    # configure image data augmentation
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_last',
        # rotation_range=15,
    )
    samples = np.expand_dims(image, 0)
    it = datagen.flow(samples, batch_size=n_examples)
    yhats = model.predict_generator(it, verbose=0)
    summed=np.sum(yhats,axis=0)
    yhat = np.argmax(summed)
    print(yhat)
    print(yhats)
    return yhat
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
def imgPrep(img, re_size=(300, 300), sigmaX=10, scale=270, prepocess=True, mask=False):
    # 图片去黑边,部分效果不好
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    center_color = binary[int(binary.shape[0] / 2), int(binary.shape[1] / 2)]
    # 有些图片背景是白色的，二值化后与黑色背景的相反
    c_x, c_y = int(img.shape[0] / 2), int(img.shape[1] / 2)
    index_x = np.argwhere(binary[:, c_y] == 255) if center_color > 128 \
        else np.argwhere(binary[:, c_y] == 0)
    half_x = int(len(index_x) / 2)
    index_y = np.argwhere(binary[c_x, :] == 255) if center_color > 128 \
        else np.argwhere(binary[c_x, :] == 0)
    half_y = int(len(index_y) / 2)
    img_c = img[np.abs(c_x - half_x):c_x + half_x, np.abs(c_y - half_y):c_y + half_y]

    if prepocess:
        # 自适应直方图均衡化
        lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 图片缩放
        x = image[image.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() // 2
        s = scale * 1.0 / r
        image = cv2.resize(image, (0, 0), fx=s, fy=s)

        # 图片叠加
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX),
                                -4, 128)
    else:
        image = img_c

    # 图片遮罩
    if mask:
        img_min = min(image.shape[0], image.shape[1])
        image = cv2.resize(image, (img_min, img_min))
        circle_img = np.zeros((img_min, img_min), np.uint8)
        cv2.circle(circle_img, (int(img_min / 2), int(img_min / 2)), int(img_min / 2), 1, thickness=-1)
        img_f = cv2.bitwise_and(image, image, mask=circle_img)
        return cv2.resize(img_f, re_size)
    else:
        return cv2.resize(image, re_size)

def new_csv(r_path, w_path):
    df = pd.DataFrame([], columns=['image_id', 'DR_Level'])
    filepaths = os.listdir(r_path)
    for filepath in filepaths:
        file = os.path.join(r_path, filepath)
        filenames = os.listdir(file)
        for filename in filenames:
            img = os.path.join(file, filename)
            # 下面这里是图像预处理 按照你们自己的来
            img1 = cv2.imread(img)
            img1 = imgPrep(img1)
            img = preprocess_image(img)
            img = img.reshape((IMG_SIZE, IMG_SIZE, 3))
            # y_pre = model.predict(img)
            # y_pre = tta_model.predict(img)
            y_pre1 = tta(img, model1)
            y_pre2 = tta(img, model2)
            y_pre3 = tta(img1, model5)
            y_pre4 = tta(img1, model6)
            y_pre5 = tta_classify(img, model3)
            y_pre6 = tta_classify(img, model4)
            y_pre = y_pre1 * 0.2 + y_pre2 * 0.2 + y_pre3 * 0.2 + y_pre4 * 0.2 + y_pre5 * 0.1 + y_pre6 * 0.1

            if (y_pre < 0.5):
                y_pre = 0
            elif (y_pre < 1.5):
                y_pre = 1
            elif (y_pre < 2.5):
                y_pre = 2
            elif (y_pre < 3.5):
                y_pre = 3
            else:
                y_pre = 4
            df = df.append(pd.DataFrame([[filename.split('.')[0], y_pre]],
                                        columns=df.columns),
                           ignore_index=True)
    df.to_csv(w_path, index=False)


# new_csv("/home/td/PycharmProjects/isbi/data/test","Challenge1_upload2.0.csv")

def load_csv(filename):
    y_pres, labels = [], []
    df = pd.DataFrame([], columns=['image_id', 'DR_Level'])
    with open(filename) as f:
        reader = csv.reader(f)
        result = list(reader)
        for row in result[1:]:
            img = '/media/td/B4DAA25FDAA21E1C/isbi' + row[2].replace('\\','/')

            # img1=cv2.imread(img)
            # img1=imgPrep(img1)
            img = preprocess_image(img)
            img = img.reshape((IMG_SIZE, IMG_SIZE, 3))
            y_pre1 = tta(img,model)
            y_pre2=tta(img,model1)
            y_pre=0.5*y_pre1+0.5*y_pre2

            # y_pre1 = tta(img, model1)
            # y_pre2 = tta(img, model2)
            # y_pre3=tta(img1,model5)
            # y_pre4=tta(img1,model6)
            # y_pre5 = tta_classify(img, model3)
            # y_pre6 = tta_classify(img, model4)
            #
            # y_pre=y_pre1*0.2+y_pre2*0.2+y_pre3*0.2+y_pre4*0.2+y_pre5*0.1+y_pre6*0.1
            # y_pre = tta_prediction(model=model, image=img, n_examples=ntta)
            if (y_pre < 0.5):
                y_pre = 0
            elif (y_pre < 1.5):
                y_pre = 1
            elif (y_pre < 2.5):
                y_pre = 2
            elif (y_pre < 3.5):
                y_pre = 3
            else:
                y_pre = 4

            if row[4] != '':
                label = row[4]
            if row[5] != '':
                label = row[5]

            label = int(label)
            print(y_pre,label)
            y_pres.append(y_pre)
            labels.append(label)
    return y_pres, labels


y_pres, y_val = load_csv(
    '/media/td/B4DAA25FDAA21E1C/isbi/regular-fundus-validation/regular-fundus-validation.csv'
)

cm = confusion_matrix(y_val, y_pres)
print(cm)
print(cm.trace()/cm.sum())
print(cohen_kappa_score(y_val, y_pres, weights='quadratic'))

# [0.4 0.3 0.2 0.1] 0.892  efficientnet 回归模型加分类模型
#[0.2 0.2 0.2 0.2 0.1 0.1] 0.895 efficientnet+InpResNetV2
#[0.3 0.1 0.3 0.1 0.1 0.1] 0.891

