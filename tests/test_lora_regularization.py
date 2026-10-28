"""Verify reference loss masking, frozen teacher, and compiled differentiability."""
from pathlib import Path
import importlib.util
import pytest
mx = pytest.importorskip('mlx.core', reason='MLX checks require Apple Silicon and the train extra')
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import default_loss

spec = importlib.util.spec_from_file_location('regularization', Path(__file__).resolve().parents[1] / 'scripts/lora_regularization.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)

    def __call__(self, tokens):
        return self.head(self.embedding(tokens))


def pair():
    mx.random.seed(42)
    student = Tiny()
    reference = Tiny()
    reference.update(student.parameters())
    mx.eval(student.parameters(), reference.parameters())
    return student, reference, mx.array([[1, 2, 3, 4, 5, 6, 0, 0]]), mx.array([[4, 6]])


def test_zero_beta_matches_existing_answer_masked_ce():
    student, teacher, batch, lengths = pair()
    actual, ntoks = module.reference_kl_loss(teacher, beta=0)(student, batch, lengths)
    expected, n = default_loss(student, batch, lengths)
    assert abs(float(actual-expected)) < 1e-6 and int(ntoks) == int(n) == 3


def test_identical_reference_has_zero_kl():
    student, teacher, batch, lengths = pair()
    actual, _ = module.reference_kl_loss(teacher, beta=2)(student, batch, lengths)
    expected, _ = default_loss(student, batch, lengths)
    assert abs(float(actual-expected)) < 1e-6
    assert not tree_flatten(teacher.trainable_parameters())


def test_reference_regularization_is_positive_and_compiles_with_gradients():
    student, teacher, batch, lengths = pair()
    # Replacing the student's head leaves the independent reference unchanged.
    before = mx.array(teacher.head.weight)
    student.head.weight = student.head.weight + mx.arange(16)[:, None]*.02
    student.reference_base = teacher
    loss = module.reference_kl_loss(teacher, beta=1)
    actual, _ = loss(student, batch, lengths)
    expected, _ = default_loss(student, batch, lengths)
    assert float(actual) > float(expected)
    value_and_grad = nn.value_and_grad(student, loss)
    mx.eval(student.parameters())
    def step(tokens, offsets):
        return value_and_grad(student, tokens, offsets)
    compiled = mx.compile(step, inputs=student.state, outputs=student.state)
    (value, ntoks), grads = compiled(batch, lengths)
    mx.eval(value, grads)
    assert bool(mx.isfinite(value)) and int(ntoks) == 3
    assert bool(mx.all(mx.isfinite(grads['head']['weight'])))
    assert bool(mx.array_equal(before, teacher.head.weight))
