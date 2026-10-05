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
"""Tests for the differentiable signed log/exp transform pair."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import transforms

# The context manager moved to the public namespace after JAX 0.7.
if hasattr(jax, 'enable_x64'):
  _enable_x64 = jax.enable_x64
else:
  _enable_x64 = jax.experimental.enable_x64


class SignedLogExpTest(parameterized.TestCase):

  @parameterized.product(
      name=('signed_logp1', 'signed_expm1'),
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64),
      compiled=(False, True),
  )
  def test_values_and_first_derivatives(self, name, dtype, compiled):
    with _enable_x64(dtype == jnp.float64):
      x = jnp.array([-3.0, -0.001, -0.0, 0.0, 0.001, 3.0], dtype=dtype)
      x64 = np.asarray(x).astype(np.float64)
      f = getattr(transforms, name)
      g = jax.vmap(jax.grad(f))
      if compiled:
        f, g = jax.jit(f), jax.jit(g)
      magnitude = np.abs(x64)
      if name == 'signed_logp1':
        expected = np.sign(x64) * np.log1p(magnitude)
        derivative = 1.0 / (1.0 + magnitude)
      else:
        expected = np.sign(x64) * np.expm1(magnitude)
        derivative = np.exp(magnitude)
      tol = {
          jnp.float16: 3e-3,
          jnp.bfloat16: 2e-2,
          jnp.float32: 2e-6,
          jnp.float64: 1e-12,
      }[dtype]
      np.testing.assert_allclose(f(x), expected, rtol=tol, atol=1e-7)
      np.testing.assert_allclose(g(x), derivative, rtol=tol)
      _, tangent = jax.jvp(f, (x,), (jnp.ones_like(x),))
      np.testing.assert_allclose(tangent, derivative, rtol=tol)
      self.assertEqual(f(x).dtype, x.dtype)
      self.assertEqual(f(x).shape, x.shape)

  @parameterized.parameters(False, True)
  def test_roundtrip_jacobian_is_identity_including_zero(self, compiled):
    def roundtrip(x):
      return transforms.signed_expm1(transforms.signed_logp1(x))

    evaluate = jax.jacrev(roundtrip)
    if compiled:
      evaluate = jax.jit(evaluate)
    x = jnp.array([-2.0, -0.0, 0.0, 2.0])
    np.testing.assert_allclose(roundtrip(x), x, rtol=2e-6)
    np.testing.assert_allclose(evaluate(x), np.eye(4), rtol=2e-6)

  @parameterized.parameters('signed_logp1', 'signed_expm1')
  def test_zero_initialized_prediction_can_receive_a_learning_signal(
      self, name
  ):
    f = getattr(transforms, name)
    derivative = jax.jit(jax.grad(lambda x: (f(x) - 1.0) ** 2))(jnp.array(0.0))
    np.testing.assert_allclose(derivative, -2.0)

  def test_nonfinite_forward_values_are_preserved(self):
    x = jnp.array([-jnp.inf, jnp.inf, jnp.nan])
    for name in ('signed_logp1', 'signed_expm1'):
      np.testing.assert_array_equal(getattr(transforms, name)(x), np.asarray(x))


if __name__ == '__main__':
  absltest.main()
