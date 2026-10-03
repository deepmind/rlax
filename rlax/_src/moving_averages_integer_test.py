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
"""Integer inputs must be promoted before forming second moments."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import moving_averages


class IntegerEmaTest(parameterized.TestCase):

  @parameterized.product(
      example=(
          (np.int8, [-100, 100]),
          (np.uint8, [0, 200]),
          (np.int16, [-1000, 1000]),
          (np.uint16, [0, 1000]),
          (np.int32, [-50000, 50000]),
          (np.uint32, [0, 100000]),
          (np.bool_, [False, True]),
      ),
      compiled=(False, True),
  )
  def test_debiased_moments_match_floating_reference(self, example, compiled):
    dtype, values = example
    batch = jnp.asarray(values, dtype=dtype)
    original = np.array(batch, copy=True)
    init, update = moving_averages.create_ema(decay=0.5)
    state = init(batch)
    apply = jax.jit(update) if compiled else update
    moments, result = apply(batch, state)
    reference = original.astype(np.float64)
    np.testing.assert_allclose(moments.mean, np.mean(reference), rtol=1e-6)
    np.testing.assert_allclose(moments.variance, np.var(reference), rtol=1e-6)
    self.assertEqual(result.nu.dtype, state.nu.dtype)
    np.testing.assert_array_equal(state.mu, 0.0)
    np.testing.assert_array_equal(state.nu, 0.0)
    np.testing.assert_array_equal(batch, original)

  def test_nested_integer_updates_remain_valid_in_a_compiled_scan(self):
    batches = {
        'signed': jnp.array(
            [[-50000, 50000], [30000, -10000], [40000, 50000]], dtype=jnp.int32
        ),
        'unsigned': jnp.array(
            [[0, 100000], [40000, 50000], [90000, 0]], dtype=jnp.uint32
        ),
    }
    init, update = moving_averages.create_ema(decay=0.5)
    state = init(jax.tree.map(lambda x: x[0], batches))

    def step(carry, batch):
      moments, next_state = update(batch, carry)
      return next_state, (moments.mean, moments.variance)

    final_state, (means, variances) = jax.jit(
        lambda s: jax.lax.scan(step, s, batches)
    )(state)
    for name, values in batches.items():
      mu = nu = 0.0
      for i, batch in enumerate(np.asarray(values, dtype=np.float64)):
        mu = 0.5 * mu + 0.5 * batch.mean()
        nu = 0.5 * nu + 0.5 * np.square(batch).mean()
        debias = 1.0 / (1.0 - 0.5 ** (i + 1))
        np.testing.assert_allclose(means[name][i], mu * debias, rtol=2e-6)
        np.testing.assert_allclose(
            variances[name][i], nu * debias - (mu * debias) ** 2, rtol=2e-6
        )
      self.assertEqual(final_state.nu[name].dtype, state.nu[name].dtype)

  def test_named_axis_mean_combines_promoted_second_moments(self):
    batches = jnp.array([[-50000, 50000], [-40000, 40000]], dtype=jnp.int32)
    init, update = moving_averages.create_ema(
        decay=0.5, pmean_axis_name='workers'
    )
    state = init(batches[0])
    moments, _ = jax.jit(
        jax.vmap(update, in_axes=(0, None), axis_name='workers')
    )(batches, state)
    np.testing.assert_allclose(moments.mean, [0.0, 0.0])
    np.testing.assert_allclose(moments.variance, [2.05e9, 2.05e9], rtol=1e-6)

  def test_float_inputs_keep_their_value_gradients(self):
    values = jnp.array([1.0, 3.0], dtype=jnp.float32)
    init, update = moving_averages.create_ema(decay=0.5)
    state = init(values)
    fn = lambda x: update(x, state)[0].variance
    self.assertAlmostEqual(float(fn(values)), 1.0)
    np.testing.assert_allclose(jax.jit(jax.grad(fn))(values), [-1.0, 1.0])
    self.assertEqual(update(values, state)[1].nu.dtype, jnp.float32)


if __name__ == '__main__':
  absltest.main()
