# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Unit tests for `transforms.py`."""

import functools
import math
from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import transforms


TWO_HOT_BINS = 5
TWO_HOT_SCALARS = [-5.0, -3.0, -1.0, -0.4, 0.0, 0.3, 1.0, 4.5, 10.0]
TWO_HOT_PROBABILITIES = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
]


class TransformsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.x = 0.5
    self.xs = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])

  @chex.all_variants()
  def test_identity_scalar(self):
    identity = self.variant(transforms.identity)  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(identity(x), self.x)

  @chex.all_variants()
  def test_identity_vector(self):
    identity = self.variant(transforms.identity)  # pyrefly: ignore[missing-attribute]
    # Test output.
    np.testing.assert_allclose(identity(self.xs), self.xs)

  @chex.all_variants()
  def test_sigmoid_scalar(self):
    sigmoid = self.variant(transforms.sigmoid)  # pyrefly: ignore[missing-attribute]
    logit = self.variant(transforms.logit)  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_sigmoid_vector(self):
    sigmoid = self.variant(transforms.sigmoid)  # pyrefly: ignore[missing-attribute]
    logit = self.variant(transforms.logit)  # pyrefly: ignore[missing-attribute]
    # Test output.
    np.testing.assert_allclose(logit(sigmoid(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_log_exp_transform_scalar(self):
    signed_logp1 = self.variant(transforms.signed_logp1)  # pyrefly: ignore[missing-attribute]
    signed_expm1 = self.variant(transforms.signed_expm1)  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(signed_expm1(signed_logp1(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_log_exp_transform_vector(self):
    signed_logp1 = self.variant(transforms.signed_logp1)  # pyrefly: ignore[missing-attribute]
    signed_expm1 = self.variant(transforms.signed_expm1)  # pyrefly: ignore[missing-attribute]
    # Test inverse.
    np.testing.assert_allclose(
        signed_expm1(signed_logp1(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_hyper_parabolic_transform_scalar(self):
    signed_hyperbolic = self.variant(transforms.signed_hyperbolic)  # pyrefly: ignore[missing-attribute]
    signed_parabolic = self.variant(transforms.signed_parabolic)  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_hyper_parabolic_transform_vector(self):
    signed_hyperbolic = self.variant(transforms.signed_hyperbolic)  # pyrefly: ignore[missing-attribute]
    signed_parabolic = self.variant(transforms.signed_parabolic)  # pyrefly: ignore[missing-attribute]
    # Test inverse.
    np.testing.assert_allclose(
        signed_parabolic(signed_hyperbolic(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_signed_power_transform_scalar(self):
    square = self.variant(functools.partial(transforms.power, p=2.))  # pyrefly: ignore[missing-attribute]
    sqrt = self.variant(functools.partial(transforms.power, p=1/2.))  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_signed_power_transform_vector(self):
    square = self.variant(functools.partial(transforms.power, p=2.))  # pyrefly: ignore[missing-attribute]
    sqrt = self.variant(functools.partial(transforms.power, p=1/2.))  # pyrefly: ignore[missing-attribute]
    # Test inverse.
    np.testing.assert_allclose(square(sqrt(self.xs)), self.xs, atol=1e-3)

  @chex.all_variants()
  def test_hyperbolic_sin_transform_scalar(self):
    sinh = self.variant(transforms.hyperbolic_sin)  # pyrefly: ignore[missing-attribute]
    arcsinh = self.variant(transforms.hyperbolic_arcsin)  # pyrefly: ignore[missing-attribute]
    x = jnp.array(self.x)
    # Test inverse.
    np.testing.assert_allclose(sinh(arcsinh(x)), self.x, atol=1e-3)
    np.testing.assert_allclose(arcsinh(sinh(x)), self.x, atol=1e-3)

  @chex.all_variants()
  def test_hyperbolic_sin_transform_vector(self):
    sinh = self.variant(transforms.hyperbolic_sin)  # pyrefly: ignore[missing-attribute]
    arcsinh = self.variant(transforms.hyperbolic_arcsin)  # pyrefly: ignore[missing-attribute]
    # Test inverse.
    np.testing.assert_allclose(sinh(arcsinh(self.xs)), self.xs, atol=1e-3)
    np.testing.assert_allclose(arcsinh(sinh(self.xs)), self.xs, atol=1e-3)

  def test_transform_to_2hot(self):
    y = transforms.transform_to_2hot(
        scalar=jnp.array(TWO_HOT_SCALARS),
        min_value=-1.0,
        max_value=1.0,
        num_bins=TWO_HOT_BINS)

    np.testing.assert_allclose(y, np.array(TWO_HOT_PROBABILITIES), atol=1e-4)

  def test_transform_from_2hot(self):
    y = transforms.transform_from_2hot(
        probs=jnp.array(TWO_HOT_PROBABILITIES),
        min_value=-1.0,
        max_value=1.0,
        num_bins=TWO_HOT_BINS)

    np.testing.assert_allclose(
        y, np.clip(np.array(TWO_HOT_SCALARS), -1, 1), atol=1e-4)

  def test_2hot_roundtrip(self):
    min_value = -1.0
    max_value = 1.0
    num_bins = 11

    value = np.arange(min_value, max_value, 0.01)

    transformed = transforms.transform_to_2hot(
        value, min_value, max_value, num_bins)
    restored = transforms.transform_from_2hot(
        transformed, min_value, max_value, num_bins)

    np.testing.assert_almost_equal(value, restored, decimal=5)


class LogitNumericsTest(parameterized.TestCase):

  @parameterized.product(dtype=(np.float32, np.float64), compiled=(False, True))
  def test_logit_values_near_probability_boundaries(self, dtype, compiled):
    if dtype == np.float64 and not jax.config.x64_enabled:
      self.skipTest('Double precision is disabled.')
    small = 1e-200 if dtype == np.float64 else 1e-30
    values = np.array(
        [
            np.finfo(dtype).tiny,
            small,
            1e-20,
            0.25,
            0.5,
            0.75,
            np.nextafter(dtype(1.0), dtype(0.0)),
        ],
        dtype=dtype,
    )
    x = jnp.asarray(values.reshape(1, -1))
    logit = jax.jit(transforms.logit) if compiled else transforms.logit
    expected = np.array(
        [math.log(float(v) / (1.0 - float(v))) for v in values], dtype=dtype
    ).reshape(x.shape)
    actual = logit(x)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0.0)
    self.assertEqual(actual.shape, x.shape)
    self.assertEqual(actual.dtype, x.dtype)
    # Check the inverse away from the normal/subnormal boundary.
    np.testing.assert_allclose(
        transforms.sigmoid(actual[:, 1:]), x[:, 1:], rtol=3e-6, atol=0.0
    )

  @parameterized.product(dtype=(np.float32, np.float64), compiled=(False, True))
  def test_logit_derivatives_remain_finite_when_representable(
      self, dtype, compiled
  ):
    if dtype == np.float64 and not jax.config.x64_enabled:
      self.skipTest('Double precision is disabled.')
    small = 1e-200 if dtype == np.float64 else 1e-30
    x = jnp.asarray(
        [small, 1e-20, 0.25, 0.5, np.nextafter(dtype(1.0), dtype(0.0))],
        dtype=dtype,
    )
    expected = np.array(
        [1.0 / (float(v) * (1.0 - float(v))) for v in np.asarray(x)]
    )
    gradient = jax.vmap(jax.grad(transforms.logit))
    if compiled:
      gradient = jax.jit(gradient)
    actual = gradient(x)
    self.assertTrue(np.all(np.isfinite(actual)))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0.0)
    forward = lambda p: jax.jvp(transforms.logit, (p,), (jnp.ones_like(p),))[1]
    if compiled:
      forward = jax.jit(forward)
    np.testing.assert_allclose(forward(x), expected, rtol=3e-6, atol=0.0)

  @parameterized.parameters(False, True)
  def test_logit_domain_endpoints(self, compiled):
    logit = jax.jit(transforms.logit) if compiled else transforms.logit
    values = jnp.asarray([-0.1, -0.0, 0.0, 0.5, 1.0, 1.1, jnp.nan])
    np.testing.assert_allclose(
        logit(values),
        [np.nan, -np.inf, -np.inf, 0.0, np.inf, np.nan, np.nan],
        equal_nan=True,
    )


if __name__ == '__main__':
  jax.config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
