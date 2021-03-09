import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import MODEL
from dataset import FaceDataset
from defaults import _C as cfg


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir_age", type=str, required=True, help="Data root directory of age")
    parser.add_argument("--data_dir_extended", type=str, required=True, help="Data root directory of extended data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device, weights):
    model.train()
    loss_monitor = AverageMeter()
    loss_age_monitor = AverageMeter()
    loss_gender_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y_age, y_gender in _tqdm:
            x = x.to(device)
            y_age = y_age.to(device)
            y_gender = y_gender.to(device)

            # compute output
            outputs_age, outputs_gender = model(x)

            # calc loss
            loss_age = criterion(outputs_age, y_age)
            loss_gender = criterion(outputs_age, y_gender)
            # add with weights
            loss = weights[0] * loss_age + weights[1] * loss_gender

            # Geometric loss
            # loss = torch.sqrt(loss_age * loss_gender)
            cur_loss = loss.item()


            # calc accuracy
            _, predicted_age = outputs_age.max(1)
            _, predicted_gender = outputs_gender.max(1)
            if weights[0]==0: ## only consider the gender_predction
              correct_num = predicted_gender.eq(y_gender).sum().item()
            elif weights[0]==1:  # only consider the age_predction
              correct_num = predicted_age.eq(y_age).sum().item()
            else:
              correct_num = (predicted_age.eq(y_age).sum().item() + predicted_gender.eq(y_gender).sum().item())/2.

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            loss_age_monitor.update(loss_age.item(), sample_num)
            loss_gender_monitor.update(loss_gender.item(), sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, loss_age_monitor.avg, loss_gender_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device, weights):
    model.eval()
    loss_monitor = AverageMeter()
    loss_age_monitor = AverageMeter()
    loss_gender_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds_age = []
    gt_age = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y_age, y_gender) in enumerate(_tqdm):
                x = x.to(device)
                y_age = y_age.to(device)
                y_gender = y_gender.to(device)

                # compute output
                outputs_age, outputs_gender = model(x)
                preds_age.append(F.softmax(outputs_age, dim=-1).cpu().numpy())

                gt_age.append(y_age.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss_age = criterion(outputs_age, y_age)
                    loss_gender = criterion(outputs_age, y_gender)
                    cur_loss = weights[0] * loss_age.item() + weights[1] * loss_gender.item()

                    # calc accuracy
                    _, predicted_age = outputs_age.max(1)
                    _, predicted_gender = outputs_gender.max(1)
                    if weights[0]==0: ## only consider the gender_predction
                      correct_num = predicted_gender.eq(y_gender).sum().item()
                    elif weights[0]==1:  # only consider the age_predction
                      correct_num = predicted_age.eq(y_age).sum().item()
                    else:
                      correct_num = (predicted_age.eq(y_age).sum().item() + predicted_gender.eq(y_gender).sum().item())/2.


                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    loss_age_monitor.update(loss_age.item(), sample_num)
                    loss_gender_monitor.update(loss_gender.item(), sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds_age = np.concatenate(preds_age, axis=0)
    gt_age = np.concatenate(gt_age, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds_age * ages).sum(axis=-1)
    diff_age = ave_preds - gt_age
    mae = np.abs(diff_age).mean() ### calcule the mae for age prediction

    return loss_monitor.avg, loss_age_monitor.avg, loss_gender_monitor.avg, accuracy_monitor.avg, mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = MODEL(model_name=cfg.MODEL.ARCH)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir_age, args.data_dir_extended, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir_age, args.data_dir_extended, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None
    weights = cfg.TRAIN.WEIGHTS_LOSS_HEAD

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_loss_age, train_loss_gender, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, weights)

        # validate
        val_loss, val_loss_age, val_loss_gender, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device, weights)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("loss_age", train_loss_age, epoch)
            train_writer.add_scalar("loss_gender", train_loss_gender, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)

            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("loss_age", val_loss_age, epoch)
            val_writer.add_scalar("loss_gender", val_loss_gender, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if weights[0] == 0:
          if epoch%10 ==0 or epoch== (cfg.TRAIN.EPOCHS -1):
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
          
        else:
          if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
          else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()