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
"""Stable categorical projection on narrowly spaced value supports."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import rlax


def reference_projection(source, weights, target):
  output = np.zeros(len(target), dtype=np.float64)
  for position, mass in zip(np.asarray(source, dtype=np.float64), weights):
    position = np.clip(position, target[0], target[-1])
    right = np.searchsorted(target, position, side='right')
    if right == len(target):
      output[-1] += mass
    else:
      left = max(0, right - 1)
      fraction = (position - target[left]) / (target[right] - target[left])
      output[left] += mass * (1 - fraction)
      output[right] += mass * fraction
  return output


class CategoricalProjectionPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32], spacing=[1e-5, 1.0, 100.0]
  )
  def test_identity_projection_keeps_probability_mass(self, dtype, spacing):
    atoms = jnp.array([0.0, spacing, 2 * spacing], dtype=dtype)
    mass = jnp.array([0.25, 0.5, 0.25], dtype=dtype)
    for fn in (
        rlax.categorical_l2_project,
        jax.jit(rlax.categorical_l2_project),
    ):
      actual = fn(atoms, mass, atoms)
      self.assertEqual(actual.dtype, mass.dtype)
      np.testing.assert_allclose(actual, mass, rtol=0.01, atol=0)
      self.assertTrue(bool(jnp.isfinite(actual).all()))

  @parameterized.parameters(jnp.float16, jnp.float32)
  def test_nonuniform_tiny_support_matches_interpolation_oracle(self, dtype):
    target = jnp.array([0.0, 1e-5, 3e-5, 4e-5], dtype)
    source = jnp.array([-1e-5, 0.5e-5, 2e-5, 5e-5], dtype)
    weights = jnp.array([0.1, 0.2, 0.3, 0.4], dtype)
    expected = reference_projection(
        np.asarray(source),
        np.asarray(weights),
        np.asarray(target, dtype=np.float64),
    )
    actual = jax.jit(rlax.categorical_l2_project)(source, weights, target)
    np.testing.assert_allclose(actual, expected, rtol=0.006, atol=1e-4)
    np.testing.assert_allclose(actual.sum(), weights.sum(), rtol=0.006)

  def test_mass_derivatives_equal_projection_of_basis_vectors(self):
    target = jnp.array([0.0, 1e-5, 3e-5], jnp.float16)
    source = jnp.array([0.5e-5, 2e-5], jnp.float16)
    weights = jnp.array([0.25, 0.75], jnp.float16)
    jacobian = jax.jit(
        jax.jacrev(lambda p: rlax.categorical_l2_project(source, p, target))
    )(weights)
    expected = np.stack(
        [
            reference_projection(
                np.asarray(source), basis, np.asarray(target, dtype=np.float64)
            )
            for basis in np.eye(2)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(jacobian, expected, rtol=0.006, atol=1e-4)

  def test_single_target_atom_retains_all_mass(self):
    actual = rlax.categorical_l2_project(
        jnp.array([-1.0, 2.0]), jnp.array([0.3, 0.7]), jnp.array([0.0])
    )
    np.testing.assert_allclose(actual, [1.0], rtol=1e-6)

  def test_public_distributional_td_loss_stays_finite(self):
    atoms = jnp.array([0.0, 1e-5, 2e-5], jnp.float16)
    predictions = jnp.array([-0.5, 0.2, 0.6], jnp.float16)
    target_logits = jnp.array([0.2, 0.3, -0.1], jnp.float16)
    fn = lambda logits: rlax.categorical_td_learning(
        atoms,
        logits,
        jnp.array(0.0, jnp.float16),
        jnp.array(1.0, jnp.float16),
        atoms,
        target_logits,
    )
    value, derivative = jax.jit(jax.value_and_grad(fn))(predictions)
    expected = -jnp.sum(
        jax.nn.softmax(target_logits.astype(jnp.float32))
        * jax.nn.log_softmax(predictions.astype(jnp.float32))
    )
    self.assertTrue(bool(jnp.isfinite(derivative).all()))
    np.testing.assert_allclose(value, expected, rtol=0.003)


if __name__ == '__main__':
  absltest.main()
