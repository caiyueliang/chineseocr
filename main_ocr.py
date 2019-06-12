# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
from PIL import Image
import model
import base64

from application import idcard
from application import trainTicket


def base64_to_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def get_image(image_path):
    # img = Image.open(image_path)                                  # PIL格式

    # 二进制方式打开图文件
    f = open(image_path, 'rb')
    # 参数image：图像base64编码
    img_base64 = base64.b64encode(f.read())     # 转base64格式
    img = base64_to_cv2(img_base64)             # 转opencv格式
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转PIL格式
    return img


# 身份证识别
def ocr_id_card(path, detect_angle=False):
    # img = cv2.imread(path)    # GBR
    img = get_image(path)       # GBR
    H, W = img.shape[:2]

    time_take = time.time()

    _, result, angle = model.model(img,
                                   path.split('/')[-1],
                                   detectAngle=detect_angle,                    # 是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,           # 字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7),       # 文本行之间测iou值
                                   leftAdjust=True,                             # 对检测的文本行进行向左延伸
                                   rightAdjust=True,                            # 对检测的文本行进行向右延伸
                                   alpha=0.01)                                  # 对检测的文本行进行向右、左延伸的倍数

    # print('[ocr_id_card] result', result)
    # print('[ocr_id_card] angle', angle)

    res = idcard.idcard(result)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

    print('[ocr_id_card] res', res)
    time_take = time.time() - time_take
    print('[ocr_id_card] time_take', time_take)


def ocr_id_card_image(image_cv, detect_angle=False):
    H, W = image_cv.shape[:2]

    time_take = time.time()

    _, result, angle = model.model(image_cv,
                                   detectAngle=detect_angle,                    # 是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,           # 字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7),       # 文本行之间测iou值
                                   leftAdjust=True,                             # 对检测的文本行进行向左延伸
                                   rightAdjust=True,                            # 对检测的文本行进行向右延伸
                                   alpha=0.01)                                  # 对检测的文本行进行向右、左延伸的倍数

    # print('[ocr_id_card] result', result)
    # print('[ocr_id_card] angle', angle)

    res = idcard.idcard(result)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

    print('[ocr_id_card] res', res)
    time_take = time.time() - time_take
    print('[ocr_id_card] time_take', time_take)

# 火车票识别
def ocr_train_ticket(path, detect_angle=False):
    img = cv2.imread(path)  # GBR
    H, W = img.shape[:2]

    time_take = time.time()

    _, result, angle = model.model(img,
                                   detectAngle=detect_angle,                    # 是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,           # 字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7),       # 文本行之间测iou值
                                   leftAdjust=True,                             # 对检测的文本行进行向左延伸
                                   rightAdjust=True,                            # 对检测的文本行进行向右延伸
                                   alpha=0.01)                                  # 对检测的文本行进行向右、左延伸的倍数

    # print('[ocr_train_ticket] result', result)
    # print('[ocr_train_ticket] angle', angle)

    res = trainTicket.trainTicket(result)
    res = res.res
    res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

    print('[ocr_train_ticket] res', res)
    time_take = time.time() - time_take
    print('[ocr_train_ticket] time_take', time_take)


if __name__ == '__main__':
    #
    ocr_id_card('/home/lijc08/1.jpeg')
    ocr_id_card('/home/lijc08/2.jpg')
    ocr_id_card('/home/lijc08/3.jpg')
    ocr_id_card('/home/lijc08/4.jpg')
    ocr_id_card('/home/lijc08/5.jpg')
    ocr_id_card('/home/lijc08/6.jpg')



