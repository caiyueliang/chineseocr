import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # print('Lstm input size', input.size())              # 1 [68, 64, 512] | 2 [68, 64, 256] | w b c
        recurrent, _ = self.rnn(input)
        # print('recurrent size', recurrent.size())           # 1 [68, 64, 512] | 2 [68, 64, 512] | w b c
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        # print('t_rec size', t_rec.size())                   # 1 [4352, 512]   | 2 [4352, 512]   | w*b c
        output = self.embedding(t_rec)                        # 全链接层 [T * b, nOut]
        # print('output size', output.size())                 # 1 [4352, 256]   | 2 [4352, 3564]  | w*b c
        output = output.view(T, b, -1)
        # print('output size', output.size())                 # 1 [68, 64, 256] | 2 [68, 64, 3564]| w b c
        return output
    

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, lstmFlag=True, init_weights=True):
        """
        是否加入lstm特征层
        """
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        self.lstmFlag = lstmFlag

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(512, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))
        else:
            self.linear = nn.Linear(nh*2, nclass)

        if init_weights:
            self._initialize_weights()

    def forward(self, input):
        # conv features
        # print('input size', input.size())               # [64, 3, 32, 270] | batch 64
        conv = self.cnn(input)
        # print('conv size', conv.size())                 # [64, 512, 1, 68] | b c h w
        b, c, h, w = conv.size()
        
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # print('conv size', conv.size())                 # [64, 512, 68] | b c w
        conv = conv.permute(2, 0, 1)                      # [w, b, c]
        # print('conv size', conv.size())                 # [68, 64, 512] | w b c

        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv)
            # print('output size', output.size())         # [68, 64, 3564] | w b c
        else:
            T, b, h = conv.size()
             
            t_rec = conv.contiguous().view(T * b, h)
             
            output = self.linear(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
