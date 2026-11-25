import torch
import torch.nn.functional as F


def _mean_pairwise_cosine(rows):
    groups = rows.shape[1]
    normalized = F.normalize(rows, dim=-1)
    cosine = normalized @ normalized.transpose(1, 2)
    return cosine.triu(diagonal=1).sum(dim=(1, 2)) / (groups * (groups - 1) / 2)


def _effective_patches(attention):
    squared_mass = attention.square().sum(dim=-1)
    return squared_mass.clamp_min(torch.finfo(attention.dtype).tiny).reciprocal()


def grouping_balance_loss(attention, min_effective_patches):
    """Penalize overlapping groups and attention supported by too few patches."""
    if attention.ndim != 3 or attention.shape[1] < 2:
        raise ValueError('Grouping attention must have shape [B, K, N] with at least two groups')
    if min_effective_patches <= 0:
        raise ValueError('The minimum effective patch count must be positive')

    with torch.autocast(device_type=attention.device.type, enabled=False):
        attention = attention.float()
        effective_patches = _effective_patches(attention)
        shortfall = (1.0 - effective_patches / min_effective_patches).clamp_min(0.0)
        balance = _mean_pairwise_cosine(attention).mean() + shortfall.square().mean()
    return balance, effective_patches.detach()


class GroupingMetrics:
    """Aggregate grouping diagnostics over images, including partial batches."""

    def __init__(self):
        self._totals = {}
        self._counts = {}

    def _add(self, name, per_image):
        self._totals[name] = self._totals.get(name, 0) + per_image.sum(dim=0, dtype=torch.float64)
        self._counts[name] = self._counts.get(name, 0) + per_image.shape[0]

    @torch.no_grad()
    def update(self, attention, body_reliability=None, descriptors=None):
        if attention.ndim != 3 or attention.shape[1] < 2:
            raise ValueError('Grouping attention must have shape [B, K, N] with at least two groups')
        batch_size, groups, patches = attention.shape

        with torch.autocast(device_type=attention.device.type, enabled=False):
            attention = attention.float()
            self._add('effective_patches', _effective_patches(attention))
            self._add('attention_cosine', _mean_pairwise_cosine(attention))

            if body_reliability is not None:
                if body_reliability.shape != (batch_size, patches):
                    raise ValueError('Body reliability must have shape [B, N] matching grouping attention')
                body_reliability = body_reliability.float()
                selected = (attention * body_reliability.unsqueeze(1)).sum(dim=-1).mean(dim=-1)
                self._add('selected_body_reliability', selected)
                self._add('base_body_reliability', body_reliability.mean(dim=-1))

            if descriptors is not None:
                if descriptors.ndim != 2 or descriptors.shape[0] != batch_size or descriptors.shape[1] % (groups + 1):
                    raise ValueError('Descriptors must contain the global and each local feature in equal-width chunks')
                local = descriptors.float().reshape(batch_size, groups + 1, -1)[:, 1:]
                self._add('local_descriptor_cosine', _mean_pairwise_cosine(local))

    def summary(self):
        return {
            'num_images': self._counts.get('effective_patches', 0),
            **{name: (total / self._counts[name]).tolist() for name, total in self._totals.items()},
        }

    def log(self, logger, prefix):
        summary = self.summary()
        if not summary['num_images']:
            return
        counts = ', '.join(f'{count:.3f}' for count in summary['effective_patches'])
        logger.info('%s - %d images, effective patches per group: [%s], attention cosine: %.6f',
                    prefix, summary['num_images'], counts, summary['attention_cosine'])
        if 'selected_body_reliability' in summary:
            selected = summary['selected_body_reliability']
            base = summary['base_body_reliability']
            logger.info('%s - selected body reliability: %.6f, mean body reliability: %.6f, lift: %+.6f',
                        prefix, selected, base, selected - base)
        if 'local_descriptor_cosine' in summary:
            logger.info('%s - local descriptor cosine: %.6f', prefix, summary['local_descriptor_cosine'])
