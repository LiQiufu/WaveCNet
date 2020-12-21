import argparse
import os, re
import random
import shutil
import warnings
import numpy as np
from datetime import datetime
from printer import Printer

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.models as models

from IPython import embed

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description = 'PyTorch ImageNet Training')
parser.add_argument('--data', metavar = 'DIR', default = '/raid/liqiufu/DATA/ILSVRC2012',
                    help = 'path to dataset')
parser.add_argument('--data_test', metavar = 'DIR', default = '/raid/liqiufu/DATA/ILSVRC2012',
                    help = 'path to dataset')
parser.add_argument('--root_save', metavar = 'DIR', default = './',
                    help = 'path to save the trained model and daily record')
parser.add_argument('-a', '--arch', metavar = 'ARCH', default = 'resnet18',
                    help = 'model architecture: ' +
                           ' | '.join(model_names) +
                           ' (default: resnet18)')
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                    help = 'number of data loading workers (default: 4)')
parser.add_argument('--epochs', default = 90, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
parser.add_argument('--start-epoch', default = 0, type = int, metavar = 'N',
                    help = 'manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default = 256, type = int,
                    metavar = 'N',
                    help = 'mini-batch size (default: 256), this is the total '
                           'batch size of all GPUs on the current node when '
                           'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default = 0.1, type = float,
                    metavar = 'LR', help = 'initial learning rate', dest = 'lr')
parser.add_argument('-ld', '--lr_decay', type=str, default = 'step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type = int, default = 30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type = int, nargs = '+', default = [125, 200, 250],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type = float, default = 0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum')
parser.add_argument('--wd', '--weight-decay', default = 1e-4, type = float,
                    metavar = 'W', help = 'weight decay (default: 1e-4)',
                    dest = 'weight_decay')

parser.add_argument('-p', '--print-freq', default = 10, type = int,
                    metavar = 'N', help = 'print frequency (default: 10)')
parser.add_argument('--pretrained', dest = 'pretrained', action = 'store_true',
                    help = 'use pre-trained model')
parser.add_argument('-r', '--resume', default = '', type = str, metavar = 'PATH',
                    help = 'path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest = 'evaluate', action = 'store_true',
                    help = 'evaluate model on validation set')
parser.add_argument('--evaluate_save', dest = 'evaluate_save', action = 'store_true',
                    help = 'save validation images off')
parser.add_argument('--seed', default = None, type = int,
                    help = 'seed for initializing training. ')
parser.add_argument('--gpu', default = None, type = str,
                    help = 'GPU id to use.')

# Added functionality from PyTorch codebase
parser.add_argument('-no', '--no_data_aug', dest = 'no_data_aug', action = 'store_true',
                    help = 'no shift-based data augmentation')
parser.add_argument('--out-dir', dest = 'out_dir', default = './', type = str,
                    help = 'output directory')
parser.add_argument('-f', '--filter_size', default = 1, type = int,
                    help = 'anti-aliasing filter size')
parser.add_argument('-w', '--wavename', default = 'haar', type = str,
                    help = 'wavename: haar, dbx, biorx.y, et al')
parser.add_argument('-es', '--evaluate_shift', dest = 'evaluate_shift', action = 'store_true',
                    help = 'evaluate model on shift-invariance')
parser.add_argument('-en', '--evaluate_noise', dest = 'evaluate_noise', action = 'store_true',
                    help = 'evaluate model on noise-invariance')
parser.add_argument('-ea', '--evaluate_accuracy', dest = 'evaluate_accuracy', action = 'store_true',
                    help = 'evaluate accuracy of the model')
parser.add_argument('--epochs_shift', default = 5, type = int, metavar = 'N',
                    help = 'number of total epochs to run for shift-invariance test')
parser.add_argument('--epochs_noise', default = 5, type = int, metavar = 'N',
                    help = 'number of total epochs to run for noise-invariance test')
parser.add_argument('-nv', '--noise_var', default = 0.1, type = float, metavar = 'N',
                    help = 'float of variance of the noise added to input for noise-invariance test')
parser.add_argument('-ba', '--batch-accum', default = 1, type = int,
                    metavar = 'N',
                    help = 'number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--embed', dest = 'embed', action = 'store_true',
                    help = 'embed statement before anything is evaluated (for debugging)')
parser.add_argument('--val-debug', dest = 'val_debug', action = 'store_true',
                    help = 'debug by training on val set')
parser.add_argument('--weights', default = None, type = str, metavar = 'PATH',
                    help = 'path to pretrained model weights')
parser.add_argument('--save_weights', default = None, type = str, metavar = 'PATH',
                    help = 'path to save model weights')
parser.add_argument('--process_name', default = None, type = str, metavar = 'name',
                    help = 'the current process object')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.process_name = '{}_{}_{}'.format(args.arch, args.filter_size, args.batch_size) if args.arch.endswith('lpf') \
        else ('{}_{}_{}'.format(args.arch, args.wavename, args.batch_size) if (
                args.arch.endswith('dwt') or args.arch.endswith('per') or args.arch.endswith('dro') or args.arch.endswith('dwt_f'))
              else '{}_original_{}'.format(args.arch, args.batch_size))

    if (not os.path.exists(os.path.join(args.out_dir, args.process_name))):
        os.mkdir(os.path.join(args.out_dir, args.process_name))
    args.save_weights = os.path.join(args.out_dir, args.process_name,
                                     args.process_name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(args.save_weights) and not (
            args.evaluate_shift or args.evaluate_save or args.evaluate_noise or args.evaluate_accuracy):
        os.mkdir(args.save_weights)
    if args.evaluate_shift:
        info_file = os.path.join(args.out_dir, args.process_name,
                                 'Eva_shift_' + args.process_name + datetime.now().strftime(
                                     '_%Y-%m-%d_%H-%M-%S_info.info'))
    elif args.evaluate_accuracy:
        info_file = os.path.join(args.out_dir, args.process_name,
                                 'Eva_accuracy_' + args.process_name + datetime.now().strftime(
                                     '_%Y-%m-%d_%H-%M-%S_info.info'))
    else:
        info_file = os.path.join(args.out_dir, args.process_name,
                                 args.process_name + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S_info.info'))
    printer = Printer(file = info_file)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        printer.pprint('You have chosen a specific GPU. Use GPU: {} for training'.format(args.gpu))
        args.gpu = [int(n) for n in re.findall(r'\d+', args.gpu)]
        args.gpu_map = {}
        for i, n in enumerate(args.gpu):
            key = 'cuda:{}'.format(n)
            value = 'cuda:{}'.format(i)
            args.gpu_map[key] = value
        args.gpu_out = 0
        args.gpu = list(range(len(args.gpu)))
        printer.pprint('args.gpu_map ==> {}'.format(args.gpu_map))
        printer.pprint('args.gpu ==> {}'.format(args.gpu))
    else:
        raise AttributeError('请设置 GPU 参数')

    main_worker(args.gpu, args, printer)


def main_worker(gpu, args, printer):
    global best_acc1
    args.gpu = gpu

    # create model
    if args.pretrained:
        printer.pprint("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained = True)
    else:
        printer.pprint("=> creating model '{}'".format(args.arch))
        import models_dwt.alexnet
        import models_dwt.vgg_v1
        import models_dwt.resnet
        import models_dwt.densenet
        import models_dwt.mobilenet
        import models_dwt_ov.resnet_A, models_dwt_ov.resnet_C

        if (args.arch == 'vgg11_bn_dwt'):
            model = models_dwt.vgg_v1.vgg11_bn(wavename = args.wavename)
        elif (args.arch == 'vgg13_bn_dwt'):
            model = models_dwt.vgg_v1.vgg13_bn(wavename = args.wavename)
        elif (args.arch == 'vgg16_bn_dwt'):
            model = models_dwt.vgg_v1.vgg16_bn(wavename = args.wavename)
        elif (args.arch == 'vgg19_bn_dwt'):
            model = models_dwt.vgg_v1.vgg19_bn(wavename = args.wavename)

        elif (args.arch == 'vgg11_dwt'):
            model = models_dwt.vgg_v1.vgg11(wavename = args.wavename)
        elif (args.arch == 'vgg13_dwt'):
            model = models_dwt.vgg_v1.vgg13(wavename = args.wavename)
        elif (args.arch == 'vgg16_dwt'):
            model = models_dwt.vgg_v1.vgg16(wavename = args.wavename)
        elif (args.arch == 'vgg19_dwt'):
            model = models_dwt.vgg_v1.vgg19(wavename = args.wavename)

        elif (args.arch == 'resnet18_dwt'):
            model = models_dwt.resnet.resnet18(wavename = args.wavename)
        elif (args.arch == 'resnet18_dwt_a'):
            model = models_dwt_ov.resnet_A.resnet18(wavename = args.wavename)
        elif (args.arch == 'resnet18_dwt_c'):
            model = models_dwt_ov.resnet_C.resnet18(wavename = args.wavename)
        elif (args.arch == 'resnet34_dwt'):
            model = models_dwt.resnet.resnet34(wavename = args.wavename)
        elif (args.arch == 'resnet50_dwt'):
            model = models_dwt.resnet.resnet50(wavename = args.wavename)
        elif (args.arch == 'resnet101_dwt'):
            model = models_dwt.resnet.resnet101(wavename = args.wavename)
        elif (args.arch == 'resnet152_dwt'):
            model = models_dwt.resnet.resnet152(wavename = args.wavename)
        elif (args.arch == 'resnext50_32x4d_dwt'):
            model = models_dwt.resnet.resnext50_32x4d(wavename = args.wavename)
        elif (args.arch == 'resnext101_32x8d_dwt'):
            model = models_dwt.resnet.resnext101_32x8d(wavename = args.wavename)

        elif (args.arch == 'densenet121_dwt'):
            model = models_dwt.densenet.densenet121(wavename = args.wavename)
        elif (args.arch == 'densenet169_dwt'):
            model = models_dwt.densenet.densenet169(wavename = args.wavename)
        elif (args.arch == 'densenet201_dwt'):
            model = models_dwt.densenet.densenet201(wavename = args.wavename)
        elif (args.arch == 'densenet161_dwt'):
            model = models_dwt.densenet.densenet161(wavename = args.wavename)

        elif (args.arch == 'mobilenet_v2_dwt'):
            model = models_dwt.mobilenet.mobilenet_v2(wavename = args.wavename)
        else:
            model = models.__dict__[args.arch]()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)


    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids = args.gpu, output_device = args.gpu_out)
        model = model.cuda()
    else:
        print('I AM HERE !!')
        model = torch.nn.DataParallel(module = model.cuda(), device_ids = args.gpu, output_device = args.gpu_out)


    if args.resume:
        if os.path.isfile(args.resume):
            printer.pprint("=> loading checkpoint '{}'".format(args.resume))
            print('map_location is {}'.format(args.gpu_map))
            #checkpoint = torch.load(args.resume, map_location = args.gpu_map)

            model_dict = model.state_dict()
            print(model_dict.keys())
            checkpoint = torch.load(args.resume, map_location = args.gpu_map)
            #pretrained_dict = [(k, v) for k, v in checkpoint['state_dict'].items() if k in model_dict]
            pretrained_dict = [(k, v) for k, v in checkpoint['state_dict'].items() if k in model_dict]
            print(checkpoint['state_dict'].keys())
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            try:
                if ('optimizer' in checkpoint.keys()):  # if no optimizer, then only load weights
                    args.start_epoch = checkpoint['epoch']
                    best_acc1 = checkpoint['best_acc1']
                    best_acc1 = best_acc1.to(args.gpu_out)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                else:
                    printer.pprint('  No optimizer saved')
                printer.pprint("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            except:
                printer.pprint('==============')
        else:
            printer.pprint("=> no checkpoint found at '{}'".format(args.resume))
    if args.weights is not None:
        printer.pprint("=> using saved weights [%s]" % args.weights)
        weights = torch.load(args.weights, map_location = args.gpu_map)
        model.load_state_dict(weights['state_dict'])
    #model = torch.nn.parallel.DataParallel(model, device_ids = args.gpu, output_device = args.gpu_out)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data_test, 'val')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean = mean, std = std)

    if (args.no_data_aug):
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle = True,
        num_workers = args.workers, pin_memory = False)

    crop_size = 256 if (args.evaluate_shift or args.evaluate_save) else 224
    args.batch_size = 1 if args.evaluate_save else args.batch_size

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size = args.batch_size // 4, shuffle = False,
        num_workers = args.workers, pin_memory = False)

    if (args.val_debug):  # debug mode - train on val set for faster epochs
        train_loader = val_loader

    if (args.embed):
        embed()

    if args.evaluate:
        validate(val_loader, model, criterion, args, printer)
        return

    if (args.evaluate_shift):
        validate_shift(val_loader, model, args, printer)
        return

    if args.evaluate_accuracy:
        validate_accuracy(val_loader, model, args, printer)
        return

    if (args.evaluate_save):
        validate_save(val_loader, mean, std, args)
        return

    start_time = datetime.now()
    validate(val_loader, model, criterion, args, printer, start_time)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, printer, start_time)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, printer, start_time)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, out_dir = args.save_weights, args = args)


def train(train_loader, model, criterion, optimizer, epoch, args, printer, start_time):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = datetime.now()

    accum_track = 0
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        # measure data loading time
        data_time.update((datetime.now() - end).total_seconds())

        if args.gpu is not None:
            input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk = (1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        accum_track += 1
        if (accum_track == args.batch_accum):
            optimizer.step()
            accum_track = 0
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update((datetime.now() - end).total_seconds())
        end = datetime.now()

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            printer.pprint('Epoch: [{0}][{1}/{2}]\t'
                           'Total_time {3}  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                           'lr {lr}'.format(
                epoch, i, len(train_loader), datetime.now() - start_time, batch_time = batch_time,
                data_time = data_time, loss = losses, top1 = top1, top5 = top5, lr = lr))


def validate(val_loader, model, criterion, args, printer, start_time = None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if start_time == None:
        start_time = datetime.now()
    with torch.no_grad():
        end = datetime.now()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk = (1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update((datetime.now() - end).total_seconds())
            end = datetime.now()

            if i % args.print_freq == 0:
                printer.pprint('Test: [{0}/{1}]\t'
                               'Total_time {2}  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\tAcc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader),
                    datetime.now() - start_time, batch_time = batch_time,
                    loss = losses, top1 = top1, top5 = top5))

        printer.pprint(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                       .format(top1 = top1, top5 = top5))

    return top1.avg


def validate_shift(val_loader, model, args, printer):
    batch_time = AverageMeter()
    consist = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start_time = datetime.now()
        end = datetime.now()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(val_loader):
                if args.gpu is not None:
                    input = input.cuda()
                    target = target.cuda()
                off0 = np.random.randint(32, size = 2)
                off1 = np.random.randint(32, size = 2)
                output0 = model(input[:, :, off0[0]:off0[0] + 224, off0[1]:off0[1] + 224])
                output1 = model(input[:, :, off1[0]:off1[0] + 224, off1[1]:off1[1] + 224])
                cur_agree = agreement(output0, output1).type(torch.FloatTensor).to(output0.device)
                consist.update(cur_agree.item(), input.size(0))

                # measure elapsed time
                batch_time.update((datetime.now() - end).total_seconds())
                end = datetime.now()
                if i % args.print_freq == 0:
                    printer.pprint('Ep [{0}/{1}]:\t'
                                   'Test: [{2}/{3}]\t'
                                   'Total_time {4}  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                   'Shift Consist {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, args.epochs_shift, i, len(val_loader),
                        datetime.now() - start_time, batch_time = batch_time, consist = consist))
        printer.pprint(' * Shift Consistency {consist.avg:.3f}'
                       .format(consist = consist))
    return consist.avg


def validate_accuracy(val_loader, model, args, printer):
    batch_time = AverageMeter()
    acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = datetime.now()
        end = datetime.now()
        for ep in range(args.epochs_noise):
            for i, (input, target) in enumerate(val_loader):
                if torch.cuda.is_available() is not None:
                    input = input.cuda()
                    target = target.cuda()
                output0 = model(input)
                cur_agree = agreement_target(output0, target).type(torch.FloatTensor).to(output0.device)
                # measure agreement and record
                acc.update(cur_agree.item(), input.size(0))
                # measure elapsed time
                batch_time.update((datetime.now() - end).total_seconds())
                end = datetime.now()
                if i % args.print_freq == 0:
                    printer.pprint('Ep [{0}/{1}]:\t'
                                   'Test: [{2}/{3}]\t'
                                   'Total_time {4}  Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                   'Acc {consist.val:.4f} ({consist.avg:.4f})\t'.format(
                        ep, args.epochs_shift, i, len(val_loader),
                        datetime.now() - start_time, batch_time = batch_time, consist = acc))
        printer.pprint(' * Accuracy Consistency {consist.avg:.3f}'
                       .format(consist = acc))
    return acc.avg


def validate_save(val_loader, mean, std, args):
    import matplotlib.pyplot as plt
    import os
    for i, (input, target) in enumerate(val_loader):
        img = (255 * np.clip(input[0, ...].data.cpu().numpy() * np.array(std)[:, None, None] + mean[:, None, None], 0,
                             1)).astype('uint8').transpose((1, 2, 0))
        plt.imsave(os.path.join(args.out_dir, '%05d.png' % i), img)


def save_checkpoint(state, is_best, epoch, out_dir, args):
    torch.save(state, os.path.join(out_dir, args.process_name + '.pth.tar'))
    if (epoch % 10 == 0):
        torch.save(state, os.path.join(out_dir, args.process_name + 'epoch_%03d.pth.tar' % epoch))
    if is_best:
        shutil.copyfile(os.path.join(out_dir, args.process_name + '.pth.tar'),
                        os.path.join(out_dir, args.process_name + '_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """

        :rtype: object
        """
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate_(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter, args):
    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((epoch - warmup_epoch) // args.step))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))
    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def agreement(output0, output1):
    pred0 = output0.argmax(dim = 1, keepdim = False)
    pred1 = output1.argmax(dim = 1, keepdim = False)
    agree = pred0.eq(pred1)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree

def agreement_target(output0, target):
    pred0 = output0.argmax(dim = 1, keepdim = False)
    agree = pred0.eq(target)
    agree = 100. * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree

if __name__ == '__main__':
    main()
