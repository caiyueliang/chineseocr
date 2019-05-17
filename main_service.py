# encoding:utf-8
import json
import cv2
import numpy as np
import base64
from logging import getLogger
import tornado.ioloop
import tornado.web
import main_ocr
from argparse import ArgumentParser


# ocr = ocr_service.get_ocr_service_instance()
logger = getLogger()


class OcrIdCardHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Ocr Hello, world")

    def post(self):
        images = self.get_argument('image', '')
        # print('[images]', images)
        image_binary = base64.b64decode(images)
        image_cv = cv2.imdecode(np.fromstring(image_binary, np.uint8), cv2.COLOR_BAYER_BG2BGR)
        # cv2.imshow('image_cv', image_cv)
        # cv2.waitKey(0)

        result = main_ocr.ocr_id_card_image(image_cv)
        self.finish(result)

    # async def post(self):
    #     # 从客户端post过来的信息中解析出图片urls
    #     urls = self.request.body.decode()
    #     urls = json.loads(urls)
    #
    #     img = await fetch_urls(urls)        # 异步下载图片url函数
    #     img = preprocessing(img)            # 预处理图片函数
    #     result = await inference(img)       # 调用tfserving预测函数
    #     result = postprocessing(result)     # result后处理函数
    #
    #     self.finish(result)  # 返回信息给客户端


def make_app():
    return tornado.web.Application([
        (r"/api/ocr/id_card", OcrIdCardHandler),
    ])


def parse_argvs():
    parser = ArgumentParser(description='ocr service')
    parser.add_argument("--port", type=int, help="port", default=9511)
    # parser.add_argument("--url", type=str, help="url", default="rtmp://127.0.0.1:1935/face/d_2")
    # parser.add_argument("--saveFaceHost", type=str, help="saveFaceHost", default="http://127.0.0.1:8000")
    # parser.add_argument("--gateId", dest="gateId", type=int, help="gateId", default=1)
    # parser.add_argument("--cameraId", dest="cameraId", type=int, help="cameraId", default=1)
    # parser.add_argument("--savePlate", dest="savePlate", type=int, help="savePlate", default=1)
    # parser.add_argument("--useVideo", type=int, help="useVideo", default=1)
    # parser.add_argument("--draw_box", type=int, help="draw plate box", default=2)
    # parser.add_argument('--rough_detect_model', type=str, help='model name', default='FaceBox')
    # parser.add_argument('--acc_detect_model', type=str, help='model name', default='resnet18')
    # parser.add_argument('--recognize_model', type=str, help='model name', default='CRNN')
    # parser.add_argument('--cuda', type=bool, help='use gpu', default=True)
    args = parser.parse_args()
    logger.info("[parse_argvs] ", args)
    print("[parse_argvs] ", args)
    return args


if __name__ == "__main__":
    args = parse_argvs()

    listen_port = args.port

    app = make_app()
    app.listen(listen_port)
    tornado.ioloop.IOLoop.current().start()
