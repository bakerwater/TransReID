import logging

import torch
import torch.nn.functional as F


@torch.inference_mode()
def do_patch_inference(model, loader, *, report_grouping=False):
    """Evaluate patch occupancy on the manifest's held-out mask split."""
    logger = logging.getLogger('transreid.test')
    device = torch.device('cuda')
    model.to(device).eval()
    statistics = torch.zeros((5, 2), dtype=torch.float64, device=device)
    grouped_body = torch.zeros(4, dtype=torch.float64, device=device)
    uniform_body = torch.zeros((), dtype=torch.float64, device=device)
    num_images = 0
    num_patches = 0
    for img, _, camid, viewid, body_mask, umbrella_mask, valid in loader:
        if not valid.all():
            raise ValueError('Held-out patch evaluation requires both mask targets for every image')
        img, camid, viewid = img.to(device), camid.to(device), viewid.to(device)
        targets = model.patch_masks_to_targets(body_mask.to(device), umbrella_mask.to(device))
        logits = model.predict_patch_reliability(img, camid, viewid)
        if logits.shape != targets.shape:
            raise ValueError(f'Patch logits {logits.shape} do not match targets {targets.shape}')

        probabilities = logits.float().sigmoid()
        predicted_positive = probabilities >= 0.5
        target_positive = targets >= 0.5
        patch_bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none')
        statistics += torch.stack([
            patch_bce.sum(dim=(0, 1), dtype=torch.float64),
            (probabilities - targets).abs().sum(dim=(0, 1), dtype=torch.float64),
            (predicted_positive & target_positive).sum(dim=(0, 1)),
            predicted_positive.sum(dim=(0, 1)),
            target_positive.sum(dim=(0, 1)),
        ])
        num_images += targets.shape[0]
        num_patches += targets.shape[0] * targets.shape[1]
        if report_grouping:
            _, attention, _ = model(img, cam_label=camid, view_label=viewid, return_grouping=True)
            body_occupancy = targets[..., 0]
            grouped_body += (attention * body_occupancy[:, None, :]).sum(dim=-1).sum(dim=0, dtype=torch.float64)
            uniform_body += body_occupancy.mean(dim=-1).sum(dtype=torch.float64)

    if not num_patches:
        raise ValueError('Held-out patch evaluation is empty')
    bce_total, absolute_error, intersection, predicted_count, target_count = statistics.cpu()
    union = predicted_count + target_count - intersection
    logger.info('Patch Reliability Results - %d images, %d patches, occupancy threshold: 0.5', num_images, num_patches)
    for index, name in enumerate(('Body', 'Umbrella')):
        logger.info(
            '%s - BCE: %.6f, MAE: %.6f, IoU: %.2f%%, F1: %.2f%%, '
            'positive target patches: %d, predicted positive patches: %d',
            name, bce_total[index] / num_patches, absolute_error[index] / num_patches,
            100 * intersection[index] / union[index].clamp_min(1),
            200 * intersection[index] / (target_count[index] + predicted_count[index]).clamp_min(1),
            target_count[index], predicted_count[index],
        )
    if report_grouping:
        counts = ', '.join(f'{value:.6f}' for value in (grouped_body / num_images).tolist())
        logger.info('Grouping held-out masks - body occupancy per group: [%s], uniform: %.6f',
                    counts, uniform_body.item() / num_images)
