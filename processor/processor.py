import logging
import os
import time
import torch
import torch.nn as nn
from loss.patch_reliability_loss import PatchReliabilityLoss
from .adaptive_grouping import GroupingMetrics, grouping_balance_loss
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

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

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    use_patch_reliability = cfg.MODEL.PATCH_RELIABILITY
    use_grouping = cfg.MODEL.ADAPTIVE_GROUPING
    if use_patch_reliability:
        patch_criterion = PatchReliabilityLoss(cfg.MODEL.PATCH_REL_UMBRELLA_WEIGHT)
        model_core = model.module if hasattr(model, 'module') else model
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        if use_patch_reliability:
            patch_bce_total = torch.zeros(2, device=device)
            num_annotated = 0
            num_annotated_batches = 0
        if use_grouping:
            group_metrics = GroupingMetrics()
            group_loss_total = torch.zeros((), device=device)
            num_grouped_images = 0
        for n_iter, batch in enumerate(train_loader):
            img, vid, target_cam, target_view = batch[:4]
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            if use_patch_reliability:
                body_mask, umbrella_mask, mask_valid = batch[4:]
                batch_annotated = int(mask_valid.sum())
                mask_valid = mask_valid.to(device)
                patch_targets = model_core.patch_masks_to_targets(
                    body_mask.to(device), umbrella_mask.to(device)
                )
            with amp.autocast(enabled=True):
                predictions = model(img, target, cam_label=target_cam, view_label=target_view)
                score, feat = predictions[:2]
                loss = loss_fn(score, feat, target, target_cam)
                if use_patch_reliability:
                    patch_loss, patch_bce = patch_criterion(predictions[2], patch_targets, mask_valid)
                    loss = loss + cfg.MODEL.PATCH_REL_LOSS_WEIGHT * patch_loss
                if use_grouping:
                    attention = predictions[3]
                    group_loss, _ = grouping_balance_loss(attention, cfg.MODEL.GROUP_MIN_EFFECTIVE_PATCHES)
                    loss = loss + cfg.MODEL.GROUP_BALANCE_WEIGHT * group_loss

            if use_patch_reliability:
                patch_bce_total += patch_bce * batch_annotated
                num_annotated += batch_annotated
                num_annotated_batches += int(batch_annotated > 0)
            if use_grouping:
                group_loss_total += group_loss.detach() * img.shape[0]
                num_grouped_images += img.shape[0]
                group_metrics.update(attention)

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

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        if use_patch_reliability:
            body_bce, umbrella_bce = (patch_bce_total / max(1, num_annotated)).tolist()
            logger.info(
                'Patch supervision - Epoch: %d, annotated samples: %d, annotated batches: %d/%d, '
                'Body BCE: %.6f, Umbrella BCE: %.6f',
                epoch, num_annotated, num_annotated_batches, n_iter + 1, body_bce, umbrella_bce,
            )
        if use_grouping:
            group_metrics.log(logger, f'Grouping train - Epoch: {epoch}')
            logger.info('Grouping balance - Epoch: %d, mean raw loss: %.6f',
                        epoch, group_loss_total.item() / num_grouped_images)
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
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
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
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
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    group_metrics = GroupingMetrics() if cfg.MODEL.ADAPTIVE_GROUPING else None

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
            if group_metrics is None:
                feat = model(img, cam_label=camids, view_label=target_view)
            else:
                feat, attention, body = model(img, cam_label=camids, view_label=target_view, return_grouping=True)
                group_metrics.update(attention, body, feat)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    if group_metrics is not None:
        group_metrics.log(logger, 'Grouping retrieval')
    return cmc[0], cmc[4]


