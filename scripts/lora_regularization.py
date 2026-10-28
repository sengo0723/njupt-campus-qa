"""Optional frozen-base KL regularization for SFT, restricted to answer tokens."""
import mlx.core as mx
import mlx.nn as nn


def reference_kl_loss(reference_model, beta=1., window=128):
    """Caller must attach reference_model as a frozen child before MLX compilation."""
    assert beta >= 0 and window > 0
    reference_model.freeze()
    reference_model.eval()

    def loss(model, batch, lengths):
        reference_model.eval()
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        steps = mx.arange(1, targets.shape[1]+1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ntoks = mask.sum()
        ce = (nn.losses.cross_entropy(logits, targets) * mask).astype(mx.float32).sum() / ntoks
        if beta == 0:
            return ce, ntoks
        # The reference is an independent, frozen copy of the official base.
        # Stop gradients explicitly; neither reference weights nor targets learn.
        teacher = mx.stop_gradient(reference_model(inputs)[:, -window:, :])
        teacher = teacher.astype(mx.float32)
        student = logits[:, -window:, :].astype(mx.float32)
        teacher_logp = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
        student_logp = student - mx.logsumexp(student, axis=-1, keepdims=True)
        kl_tokens = (mx.exp(teacher_logp) * (teacher_logp-student_logp)).sum(axis=-1)
        kl_mask = mask[:, -window:]
        kl = (kl_tokens * kl_mask).sum() / mx.maximum(kl_mask.sum(), 1)
        return ce + beta*mx.maximum(kl, 0.), ntoks

    return loss
