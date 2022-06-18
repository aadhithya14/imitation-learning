"""Tests `imitation.util.networks`."""

import functools
import math
from typing import Type

import pytest
import torch as th

from imitation.util import networks

assert_equal = functools.partial(th.testing.assert_close, rtol=0, atol=0)


NORMALIZATION_LAYERS = [networks.RunningNorm, networks.EMANorm]


@pytest.mark.parametrize("normalization_layer", [networks.RunningNorm])
def test_running_norm_identity(normalization_layer: Type[networks.BaseNorm]) -> None:
    """Tests running norm starts and stays at identity function.

    Specifically, we test in evaluation mode (initializatn should not change)
    and in training mode with already normalized data.

    Args:
        normalization_layer: the normalization layer to be tested.
    """
    running_norm = normalization_layer(1, eps=0.0)
    x = th.Tensor([-1.0, 0.0, 7.32, 42.0])
    running_norm.eval()  # stats should not change in eval mode
    for i in range(10):
        assert_equal(running_norm.forward(x), x)
    running_norm.train()  # stats will change in train mode
    normalized = th.Tensor([-1, 1])  # mean 0, variance 1
    for i in range(10):
        assert_equal(running_norm.forward(normalized), normalized)


@pytest.mark.parametrize("normalization_layer", NORMALIZATION_LAYERS)
def test_running_norm_eval_fixed(
    normalization_layer: Type[networks.BaseNorm],
    batch_size: int = 8,
    num_batches: int = 10,
    num_features: int = 4,
) -> None:
    """Tests that stats do not change when in eval mode and do when in training."""
    running_norm = normalization_layer(num_features)

    def do_forward(shift: float = 0.0, scale: float = 1.0):
        for i in range(num_batches):
            data = th.rand(batch_size, num_features) * scale + shift
            running_norm.forward(data)

    with th.random.fork_rng():
        th.random.manual_seed(42)

        do_forward()
        current_mean = th.clone(running_norm.running_mean)
        current_var = th.clone(running_norm.running_var)

        running_norm.eval()
        do_forward()
        assert_equal(running_norm.running_mean, current_mean)
        assert_equal(running_norm.running_var, current_var)

        running_norm.train()
        do_forward(1.0, 2.0)
        assert th.all((running_norm.running_mean - current_mean).abs() > 0.01)
        assert th.all((running_norm.running_var - current_var).abs() > 0.01)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_running_norm_matches_dist(batch_size: int) -> None:
    """Test running norm converges to empirical distribution."""
    mean = th.Tensor([-1.3, 0.0, 42])
    var = th.Tensor([0.1, 1.0, 42])
    sd = th.sqrt(var)

    num_dims = len(mean)
    running_norm = networks.RunningNorm(num_dims)
    running_norm.train()

    num_samples = 256
    with th.random.fork_rng():
        th.random.manual_seed(42)
        data = th.randn(num_samples, num_dims) * sd + mean
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            running_norm.forward(batch)

    empirical_mean = th.mean(data, dim=0)
    empirical_var = th.var(data, dim=0, unbiased=False)

    normalized = th.Tensor([[-1.0], [0.0], [1.0], [42.0]])
    normalized = th.tile(normalized, (1, 3))
    scaled = normalized * th.sqrt(empirical_var + running_norm.eps) + empirical_mean
    running_norm.eval()
    for i in range(5):
        th.testing.assert_close(running_norm.forward(scaled), normalized)

    # Stats should match empirical mean (and be unchanged by eval)
    th.testing.assert_close(running_norm.running_mean, empirical_mean)
    th.testing.assert_close(running_norm.running_var, empirical_var)
    assert running_norm.count == num_samples


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("alpha", [0.01, 0.5, 0.9, 0.98, 0.99])
def test_running_norm_matches_exactly(batch_size: int, alpha: float) -> None:
    """Test running norm matches exactly with to empirical statistics."""
    running_norm = networks.EMANorm(1, alpha=alpha)
    running_norm.train()

    data = th.as_tensor([float(i) for i in range(9)] * batch_size).reshape(-1, 1)
    num_samples = len(data)
    num_batches = num_samples // batch_size

    batch_means, batch_weights, weights = [], [], []

    for i in range(1, num_batches + 1):
        _weights = []
        if i == 1:
            # The weight for the first element = alpha ** (n - 1)
            _weights = [(1 - alpha) ** (num_batches - 1)]
        else:
            # The weights for following elements = alpha ** (n - i) * (1 - alpha)
            _weights.append((1 - alpha) ** (num_batches - i) * alpha)
        weights += _weights * batch_size
        batch_weights += _weights

    weights = th.as_tensor(weights).view(-1, 1) / batch_size
    batch_weights = th.as_tensor(batch_weights).view(-1, 1)

    th.testing.assert_close(
        weights.sum(),
        th.tensor(1.0),
    )  # assert the sum of weights is 1.0
    assert len(weights) == num_samples == len(batch_weights) * batch_size

    with th.random.fork_rng():
        th.random.manual_seed(42)
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            batch_means.append(th.mean(batch, dim=0))
            running_norm.forward(batch)
        batch_means = th.cat(batch_means)
    batch_means = batch_means.view(-1, 1)

    empirical_mean = th.mul(weights, data).sum(dim=0)
    empirical_var = th.mul(weights, th.square(data - empirical_mean)).sum(dim=0)
    empirical_var_batch_means = th.mul(
        batch_weights,
        th.square(batch_means - empirical_mean),
    ).sum(dim=0)

    running_norm.eval()

    # Stats should match empirical mean (and be unchanged by eval)
    print(running_norm.running_mean, empirical_mean)
    print(running_norm.running_var, empirical_var)
    print(running_norm.running_var, empirical_var_batch_means)
    th.testing.assert_close(running_norm.running_mean, empirical_mean)
    # th.testing.assert_close(running_norm.running_var, empirical_var)
    th.testing.assert_close(running_norm.running_var, empirical_var_batch_means)

    assert running_norm.count == num_samples


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("normalization_layer", NORMALIZATION_LAYERS)
def test_parameters_converge(
    batch_size: int,
    normalization_layer: Type[networks.BaseNorm],
) -> None:
    """Test running norm parameters approximately converge to true values."""
    mean = th.Tensor([42, 1])
    var = th.Tensor([42, 1])
    sd = th.sqrt(var)

    num_dims = len(mean)
    if normalization_layer is networks.RunningNorm:
        running_norm = normalization_layer(num_dims)
    elif normalization_layer is networks.EMANorm:
        running_norm = normalization_layer(num_dims, alpha=0.9)
    else:
        raise ValueError("Unknown normalization layer")
    running_norm.train()

    num_samples = 1000
    with th.random.fork_rng():
        th.random.manual_seed(42)
        data = th.randn(num_samples, num_dims) * sd + mean
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            running_norm.forward(batch)

    running_norm.eval()

    print(running_norm.running_mean, mean)
    print(running_norm.running_var, var)
    th.testing.assert_close(running_norm.running_mean, mean, rtol=0.03, atol=10)
    th.testing.assert_close(running_norm.running_var, var, rtol=0.1, atol=10)

    assert running_norm.count == num_samples


@pytest.mark.parametrize(
    "init_kwargs",
    [{}] + [{"normalize_input_layer": layer} for layer in NORMALIZATION_LAYERS],
)
def test_build_mlp_norm_training(init_kwargs) -> None:
    """Tests MLP building function `networks.build_mlp()`.

    Specifically, we initialize an MLP and train it on a toy task. We also test the
    init options of input layer normalization.

    Args:
        init_kwargs: dict of kwargs to pass to `networks.build_mlp()`.
    """
    # Create Tensors to hold input and outputs.
    x = th.linspace(-math.pi, math.pi, 200).reshape(-1, 1)
    y = th.sin(x)
    # Construct our model by instantiating the class defined above
    model = networks.build_mlp(in_size=1, hid_sizes=[16, 16], out_size=1, **init_kwargs)

    # Construct a loss function and an Optimizer.
    criterion = th.nn.MSELoss(reduction="sum")
    optimizer = th.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(200):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
