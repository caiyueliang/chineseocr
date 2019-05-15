# -*- coding: utf-8 -*-
import cv2
import time
from PIL import Image
import model

from application import idcard


def ocr(path, textLine=False, textAngle=False, billModel='身份证'):
    img = cv2.imread(path)  # GBR
    H, W = img.shape[:2]

    time_take = time.time()

    if textLine:
        # 单行识别
        partImg = Image.fromarray(img)
        text = model.crnnOcr(partImg.convert('L'))
        res = [{'text': text, 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]
    else:
        detectAngle = textAngle  # 是否进行文字方向检测
        _, result, angle = model.model(img,
                                       detectAngle=detectAngle,  # 是否进行文字方向检测，通过web传参控制
                                       config=dict(MAX_HORIZONTAL_GAP=50,  # 字符之间的最大间隔，用于文本行的合并
                                                   MIN_V_OVERLAPS=0.6,
                                                   MIN_SIZE_SIM=0.6,
                                                   TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                   TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                   TEXT_LINE_NMS_THRESH=0.7),  # 文本行之间测iou值
                                       leftAdjust=True,  # 对检测的文本行进行向左延伸
                                       rightAdjust=True,  # 对检测的文本行进行向右延伸
                                       alpha=0.01)  # 对检测的文本行进行向右、左延伸的倍数

        print('[ocr] result', result)
        print('[ocr] angle', angle)

        # if billModel == '' or billModel == '通用OCR':
        #     result = union_rbox(result, 0.2)
        #     res = [{'text': x['text'],
        #             'name': str(i),
        #             'box': {'cx': x['cx'],
        #                     'cy': x['cy'],
        #                     'w': x['w'],
        #                     'h': x['h'],
        #                     'angle': x['degree']
        #                     }
        #             } for i, x in enumerate(result)]
        #     res = adjust_box_to_origin(img, angle, res)  # 修正box
        # elif billModel == '火车票':
        #     res = trainTicket.trainTicket(result)
        #     res = res.res
        #     res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
        # elif billModel == '身份证':
        if billModel == '身份证':
            res = idcard.idcard(result)
            res = res.res
            res = [{'text': res[key], 'name': key, 'box': {}} for key in res]

    print('[ocr] res', res)
    time_take = time.time() - time_take
    print('[ocr] time_take', time_take)


if __name__ == '__main__':
    ocr('/home/lijc08/1015929936.jpg')
