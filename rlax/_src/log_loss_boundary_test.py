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
"""Accurate binary log losses and finite endpoint prediction derivatives."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special
import rlax


class LogLossBoundaryTest(parameterized.TestCase):

  @parameterized.parameters((0.0, 0.0, 1.0), (1.0, 1.0, -1.0))
  def test_correct_endpoint_has_finite_first_and_second_derivatives(
      self, p, y, grad
  ):
    objective = lambda x: rlax.log_loss(x, jnp.array(y))
    for value_and_grad in (
        jax.value_and_grad(objective),
        jax.jit(jax.value_and_grad(objective)),
    ):
      value, actual_grad = value_and_grad(jnp.array(p))
      self.assertEqual(float(value), 0.0)
      self.assertEqual(float(actual_grad), grad)
    self.assertEqual(
        float(jax.jit(jax.grad(jax.grad(objective)))(jnp.array(p))), 1.0
    )

  @parameterized.parameters(1e-5, 1e-8, 1e-12)
  def test_small_negative_class_probability_preserves_nonzero_loss(self, p):
    value = jnp.array(p, jnp.float32)
    actual = jax.jit(rlax.log_loss)(value, jnp.array(0.0))
    np.testing.assert_allclose(
        actual, -np.log1p(-float(value)), rtol=2e-6, atol=0
    )
    self.assertGreater(float(actual), 0.0)

  @parameterized.parameters(0.0, 0.25, 0.8, 1.0)
  def test_interior_values_and_prediction_gradients_match_binary_cross_entropy(
      self, label
  ):
    predictions = jnp.array([1e-5, 0.2, 0.6, 1.0 - 1e-5], jnp.float32)
    target = jnp.array(label, jnp.float32)
    p = np.asarray(predictions, dtype=np.float64)
    expected = -special.xlogy(label, p) - special.xlog1py(1 - label, -p)
    objective = lambda x: rlax.log_loss(x, target).sum()
    value, grad = jax.jit(jax.value_and_grad(objective))(predictions)
    np.testing.assert_allclose(value, expected.sum(), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(
        grad, (1 - label) / (1 - p) - label / p, rtol=2e-6, atol=1e-6
    )

  def test_soft_label_derivatives_and_broadcasting_are_preserved(self):
    predictions = jnp.array([[0.2], [0.8]])
    targets = jnp.array([[0.0, 0.3, 1.0]])
    actual = rlax.log_loss(predictions, targets)
    p, y = np.broadcast_arrays(
        np.asarray(predictions, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )
    np.testing.assert_allclose(
        actual, -special.xlogy(y, p) - special.xlog1py(1 - y, -p), rtol=2e-6
    )
    gradient = jax.grad(lambda y: rlax.log_loss(predictions, y).sum())(targets)
    np.testing.assert_allclose(
        gradient,
        (np.log1p(-p) - np.log(p)).sum(axis=0, keepdims=True),
        atol=5e-7,
    )

  def test_confident_wrong_predictions_remain_infinite(self):
    result = rlax.log_loss(jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]))
    self.assertTrue(bool(jnp.isposinf(result).all()))

  def test_saturated_sigmoid_model_has_finite_parameter_gradients(self):
    logits = jnp.array([-100.0, 100.0, 0.0])
    labels = jnp.array([0.0, 1.0, 1.0])
    objective = lambda x: rlax.log_loss(jax.nn.sigmoid(x), labels).sum()
    value, grad = jax.jit(jax.value_and_grad(objective))(logits)
    np.testing.assert_allclose(value, np.log(2.0), rtol=2e-6)
    np.testing.assert_array_equal(grad, jnp.array([0.0, 0.0, -0.5]))
    updated = logits - 0.1 * grad
    self.assertLess(float(objective(updated)), float(value))

  def test_masked_perfect_predictions_do_not_poison_other_gradients(self):
    p = jnp.array([0.0, 1.0, 0.25])
    labels = jnp.array([0.0, 1.0, 1.0])
    mask = jnp.array([0.0, 0.0, 1.0])
    grad = jax.grad(lambda x: jnp.sum(mask * rlax.log_loss(x, labels)))(p)
    np.testing.assert_array_equal(grad, [0.0, 0.0, -4.0])


if __name__ == '__main__':
  absltest.main()
