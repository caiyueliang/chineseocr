# coding:utf-8
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torch.autograd import Variable 
from crnn.utils import strLabelConverter, resizeNormalize
from crnn.network_torch import CRNN
from crnn import keys
from collections import OrderedDict
from config import ocrModel, LSTMFLAG, GPU
from config import chinsesModel
from PIL import Image


def crnnSource():
    """
    加载模型
    """
    if chinsesModel:
        # alphabet = keys.alphabetChinese             # 中英文模型
        alphabet = keys.alphabetChinese_3564        # 中英文模型
    else:
        alphabet = keys.alphabetEnglish             # 英文模型
        
    converter = strLabelConverter(alphabet)
    # if torch.cuda.is_available() and GPU:
    #     model = CRNN(32, 3, len(alphabet)+1, 256, 1, lstmFlag=LSTMFLAG).cuda()      # LSTMFLAG=True crnn 否则 dense ocr
    # else:
    #     model = CRNN(32, 3, len(alphabet)+1, 256, 1, lstmFlag=LSTMFLAG).cpu()
    if torch.cuda.is_available() and GPU:
        model = CRNN(32, 1, len(alphabet)+1, 256, 1, lstmFlag=LSTMFLAG).cuda()      # LSTMFLAG=True crnn 否则 dense ocr
    else:
        model = CRNN(32, 1, len(alphabet)+1, 256, 1, lstmFlag=LSTMFLAG).cpu()

    trainWeights = torch.load(ocrModel, map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.', '') # remove `module.`
        modelWeights[name] = v
    # load params
  
    model.load_state_dict(modelWeights)

    return model, converter


# 加载模型
model, converter = crnnSource()
model.eval()

transform = T.Compose([
    # T.Resize((self.img_h, self.img_w)),
    T.ToTensor(),
    # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


# def crnnOcr(image):
#     """
#     crnn模型，ocr识别
#     image:PIL.Image.convert("L")
#     """
#     # print(image.size)
#     scale = image.size[1] * 1.0 / 32
#     w = image.size[0] / scale
#     w = int(w)
#
#     # image = image.resize((w, 32), resample=Image.HAMMING)
#     image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     print(image_cv.shape)
#     image_cv = cv2.resize(image_cv, (w, 32))
#     cv2.imshow("crnnOcr", image_cv)
#     image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
#
#     image = transform(image)
#
#     if torch.cuda.is_available() and GPU:
#         image = image.cuda()
#     else:
#         image = image.cpu()
#
#     print(image.size())
#     image = image.view(1, *image.size())
#     print(image.size())
#     image = Variable(image)
#     preds = model(image)
#     _, preds = preds.max(2)
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#     sim_pred = converter.decode(preds)
#     print('sim_pred', sim_pred)
#
#     cv2.waitKey(0)
#     return sim_pred


def crnnOcr(image):
    """
    crnn模型，ocr识别
    image:PIL.Image.convert("L")
    """
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # transformer = resizeNormalize((w, 32))
    # image = transformer(image)
    # image = image.astype(np.float32)
    # image = torch.from_numpy(image)
    image = image.resize((w, 32), Image.HAMMING)

    image_cv = cv2.cvtColor(np.asarray(image), cv2.COLOR_GRAY2RGB)
    cv2.imshow("crnnOcr", image_cv)

    image = transform(image)

    if torch.cuda.is_available() and GPU:
        image = image.cuda()
    else:
        image = image.cpu()

    image = image.view(1, *image.size())
    print(image.size())

    image = Variable(image)
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_pred = converter.decode(preds)
    print('sim_pred', sim_pred)

    cv2.waitKey(0)
    return sim_pred, w
