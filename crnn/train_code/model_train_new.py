# encoding:utf-8
from __future__ import print_function
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms as T
# from warpctc_pytorch import CTCLoss
import os
import train_code.utils as utils
import train_code.my_dataset as my_dataset
import numpy as np
import time


class ModuleTrain:
    def __init__(self, train_path, test_path, model_file, model, num_class_new, alphabet,
                 fine_tuning=False, img_h=32, img_w=110, batch_size=64, lr=1e-3,
                 use_unicode=True, best_loss=10, use_gpu=True, workers=1):
        self.model = model
        self.model_file = model_file
        self.use_unicode = use_unicode
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.lr = lr
        self.best_loss = best_loss
        self.best_acc = 0.95
        self.use_gpu = use_gpu
        self.workers = workers

        self.converter = utils.strLabelConverter(alphabet)
        # self.criterion = CTCLoss()
        self.criterion = torch.nn.CTCLoss()

        if self.use_gpu:
            print("[use gpu] ...")
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        if torch.cuda.is_available() and not self.use_gpu:
            print("[WARNING] You have a CUDA device, so you should probably run with --cuda")

        # 加载模型
        if os.path.exists(self.model_file):
            self.load(self.model_file)
        else:
            print('[Load model] error !!!')

        self.transform = T.Compose([
            T.Resize((self.img_h, self.img_w)),
            T.ToTensor(),
            # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        # train_label = os.path.join(train_path, 'labels_normal.txt')
        train_dataset = my_dataset.MyDataset(root=train_path, transform=self.transform,
                                             is_train=True, img_h=self.img_h, img_w=self.img_w)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                        shuffle=False, num_workers=int(self.workers))
        # test_label = os.path.join(test_path, 'labels_normal.txt')
        test_dataset = my_dataset.MyDataset(root=test_path, transform=self.transform,
                                            is_train=False, img_h=self.img_h, img_w=self.img_w)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                       shuffle=False, num_workers=int(self.workers))

        # setup optimizer
        # if opt.adam:
        #     self.optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # elif opt.adadelta:
        #     self.optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
        # else:
        #     self.optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)

        if fine_tuning:
            print(self.model)
            in_features = self.model.rnn[1].embedding.in_features                       # 提取fc层中固定的输入参数
            self.model.rnn[1].embedding = torch.nn.Linear(in_features, num_class_new)   # 修改类别为num_classes
            print(self.model)
            if self.use_gpu:
                print("[use gpu] ...")
                self.model = self.model.cuda()

    def train(self, epoch, decay_epoch=80):
        image = torch.FloatTensor(self.batch_size, 3, self.img_h, self.img_w)
        text = torch.IntTensor(self.batch_size * 5)
        length = torch.IntTensor(self.batch_size)
        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

        print('[train] epoch: %d' % epoch)
        for epoch_i in range(epoch):
            start_time = time.time()
            train_loss = 0.0
            correct = 0

            if epoch_i >= decay_epoch and epoch_i % decay_epoch == 0:                   # 减小学习速率
                self.lr = self.lr * 0.1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

            print('================================================')
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):              # 训练
                # data, target = Variable(data), Variable(target)

                if self.use_unicode:
                    # target = [tx.decode('utf-8') for tx in target]
                    target = [tx for tx in target]
                    # print(target)

                batch_size = data.size(0)
                utils.loadData(image, data)
                t, l = self.converter.encode(target)
                # print(t)
                # print(l)
                utils.loadData(text, t)
                utils.loadData(length, l)

                if self.use_gpu:
                    image = image.cuda()

                # 梯度清0
                self.optimizer.zero_grad()
                for p in self.model.parameters():
                    p.requires_grad = True

                # 计算损失
                preds = self.model(image)
                preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                # print('preds_size', preds_size)
                loss = self.criterion(preds, text, preds_size, length)
                # self.model.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新参数
                self.optimizer.step()
                train_loss += loss.item()
                if np.isnan(loss.item()):
                    print(loss.item())
                    print(target)
                    print(t)
                    print(l)
                # print(preds.size())
                # total = 0.0
                # print('len', len(preds.data[0][0]))
                # for i in range(len(preds.data[0][0])):
                #     total += preds.data[0][0][i]
                #     print('total', total)

                _, preds = preds.max(2)
                # print(preds.size())
                # preds = preds.squeeze(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                # print(preds.size())
                sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
                # print(sim_preds)
                # print(target)
                # total_preds = self.converter.decode(preds.data, preds_size.data, raw=True)
                # print(total_preds)
                for pred, target in zip(sim_preds, target):
                    # print('pred', pred, type(pred))
                    # print('target', target, type(target))
                    if pred.strip() == target.strip():
                        correct += 1

            train_loss /= len(self.train_loader)
            acc = float(correct) / float(len(self.train_loader.dataset))
            use_time = time.time() - start_time
            print('[Train] Epoch: {} \tLoss: {:.6f}\tAcc: {:.6f}\tlr: {}\ttime: {}'.format(epoch_i, train_loss, acc, self.lr, use_time))

            # Test
            test_loss, test_acc = self.test()
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                str_list = self.model_file.split('.')
                best_model_file = ""
                for str_index in range(len(str_list)):
                    best_model_file = best_model_file + str_list[str_index]
                    if str_index == (len(str_list) - 2):
                        best_model_file += '_best'
                    if str_index != (len(str_list) - 1):
                        best_model_file += '.'
                self.save(best_model_file)  # 保存最好的模型

            if test_acc > self.best_acc:
                self.best_acc = test_acc
                str_list = self.model_file.split('.')
                best_model_file = ""
                for str_index in range(len(str_list)):
                    best_model_file = best_model_file + str_list[str_index]
                    if str_index == (len(str_list) - 2):
                        best_model_file += '_best_acc'
                    if str_index != (len(str_list) - 1):
                        best_model_file += '.'
                self.save(best_model_file)  # 保存最好的模型

        self.save(self.model_file)

    def test(self):
        image = torch.FloatTensor(self.batch_size, 3, self.img_h, self.img_w)
        text = torch.IntTensor(self.batch_size * 5)
        length = torch.IntTensor(self.batch_size)
        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

        for p in self.model.parameters():
            p.requires_grad = False

        test_loss = 0.0
        correct = 0
        # loss_avg = utils.averager()

        time_start = time.time()
        self.model.eval()
        for data, target in self.test_loader:
            cpu_images = data
            cpu_texts = target
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            if self.use_unicode:
                # cpu_texts = [tx.decode('utf-8') for tx in cpu_texts]
                cpu_texts = [tx for tx in cpu_texts]

            t, l = self.converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            if self.use_gpu:
                image = image.cuda()

            preds = self.model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            loss = self.criterion(preds, text, preds_size, length)
            test_loss += loss.item()

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred.strip() == target.strip():
                    correct += 1
                # else:
                #     print(pred.strip())
                #     print(target.strip())

        time_end = time.time()
        time_avg = float(time_end - time_start) / float(len(self.test_loader.dataset))
        accuracy = correct / float(len(self.test_loader.dataset))
        test_loss /= len(self.test_loader)
        print('[Test] loss: %f, accuray: %f, time: %f' % (test_loss, accuracy, time_avg))
        return test_loss, accuracy

    def load(self, name):
        print('[Load model] %s ...' % name)
        self.model.load_state_dict(torch.load(name))
        # self.model.load(name)

    def save(self, name):
        print('[Save model] %s ...' % name)
        torch.save(self.model.state_dict(), name)
        # self.model.save(name)
