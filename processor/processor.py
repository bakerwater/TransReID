import logging
import json
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


def save_metrics(cfg, cmc, mean_ap, epoch=None):
    if not cfg.OUTPUT_DIR:
        return
    result = {'dataset': cfg.DATASETS.NAMES, 'epoch': epoch,
              'mAP': float(mean_ap), 'rank1': float(cmc[0]),
              'rank5': float(cmc[min(4, len(cmc) - 1)]),
              'rank10': float(cmc[min(9, len(cmc) - 1)]),
              'reranking': False}
    name = 'metrics.json' if epoch is None else 'metrics_epoch_{}.json'.format(epoch)
    path = os.path.join(cfg.OUTPUT_DIR, name)
    with open(path + '.tmp', 'w') as handle:
        json.dump(result, handle, indent=2)
    os.replace(path + '.tmp', path)


def get_grl_alpha(epoch, iteration, total_iters, total_epochs,
                  warmup_epochs=5, max_alpha=1.0):
    """DANN-style logistic GRL scheduling after the warmup epochs."""
    current_epoch = (epoch - 1) + (iteration + 1) / total_iters
    progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    progress = max(0.0, min(1.0, progress))
    alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
    return max_alpha * alpha


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    use_grl = bool(getattr(cfg.MODEL, 'CAMERA_GRL', False))

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    camera_acc_meter = AverageMeter()
    camera_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    start_epoch = 1
    if cfg.SOLVER.RESUME:
        state = torch.load(cfg.SOLVER.RESUME, map_location=device, weights_only=False)
        target_model = model.module if hasattr(model, 'module') else model
        target_model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        optimizer_center.load_state_dict(state['optimizer_center'])
        center_criterion.load_state_dict(state['center_criterion'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        start_epoch = state['epoch'] + 1
        logger.info('Resumed training at epoch %s', start_epoch)
        del state
    # train
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        camera_acc_meter.reset()
        camera_loss_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if use_grl:
                grl_alpha = get_grl_alpha(
                    epoch=epoch, iteration=n_iter, total_iters=len(train_loader),
                    total_epochs=epochs,
                    warmup_epochs=int(getattr(cfg.MODEL, 'GRL_WARMUP_EPOCHS', 5)),
                    max_alpha=float(getattr(cfg.MODEL, 'GRL_MAX_ALPHA', 1.0)))
            else:
                grl_alpha = 1.0
            with amp.autocast(enabled=True):
                if use_grl:
                    model_outputs = model(img, target, cam_label=target_cam,
                                          view_label=target_view, grl_alpha=grl_alpha)
                else:
                    model_outputs = model(img, target, cam_label=target_cam, view_label=target_view)
                if len(model_outputs) == 3:
                    score, feat, camera_score = model_outputs
                    loss = loss_fn(score, feat, target, target_cam, camera_score=camera_score)
                else:
                    score, feat = model_outputs
                    camera_score = None
                    loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            if camera_score is not None:
                # Monitor the unweighted camera CE without adding a second loss.
                camera_ce = F.cross_entropy(camera_score.detach().float(), target_cam)
                camera_loss_meter.update(camera_ce.item(), img.shape[0])
                camera_acc = (camera_score.argmax(1) == target_cam).float().mean()
                camera_acc_meter.update(camera_acc.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if camera_score is not None and use_grl:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID-Acc: {:.3f}, "
                                "Adv-CE: {:.3f}, AdvHead-Acc: {:.3f}, GRL-alpha: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, camera_loss_meter.avg,
                                        camera_acc_meter.avg, grl_alpha, scheduler._get_lr(epoch)[0]))
                elif camera_score is not None:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Cam-Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, camera_acc_meter.avg,
                                        scheduler._get_lr(epoch)[0]))
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0 or epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0 or epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute(return_distmat=False)
                    save_metrics(cfg, cmc, mAP, epoch)
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute(return_distmat=False)
                save_metrics(cfg, cmc, mAP, epoch)
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

        # Save RNG state after validation, whose DataLoader also consumes RNG state.
        if (epoch % checkpoint_period == 0 or epoch == epochs) and (
                not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
            target_model = model.module if hasattr(model, 'module') else model
            state = {'epoch': epoch, 'model': target_model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'optimizer_center': optimizer_center.state_dict(),
                     'center_criterion': center_criterion.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all(),
                     'config': cfg.dump()}
            path = os.path.join(cfg.OUTPUT_DIR, 'latest_training.pth')
            torch.save(state, path + '.tmp')
            os.replace(path + '.tmp', path)
            del state


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute(return_distmat=False)
    save_metrics(cfg, cmc, mAP)
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


