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
"""Intrinsic rewards use squared distances, not negative search scores."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from rlax._src import exploration


def _reference(embeddings, memory, neighbors, total=0.0, count=0, scale=1.0):
  distances = np.sum(
      (np.asarray(embeddings)[:, None, :] - np.asarray(memory)[None, :, :])
      ** 2,
      axis=-1,
  )
  nearest = np.sort(distances, axis=-1)[:, :neighbors]
  total = total + np.sum(nearest)
  count = count + nearest.size
  rate = np.maximum(nearest / (total / count + 1e-3) - 8e-3, 0.0)
  similarity = np.sqrt(np.sum(1e-4 / (rate + 1e-4), axis=-1)) + 1e-3
  return scale * np.where(similarity > 8.0, 0.0, 1.0 / similarity), total, count


class IntrinsicRewardDistanceTest(parameterized.TestCase):

  @parameterized.product(
      offset=(0.01, 0.1, 1.0), neighbors=(1, 2), jit=(False, True)
  )
  def test_initial_rewards_match_squared_distance_reference(
      self, offset, neighbors, jit
  ):
    embeddings = jnp.array([[offset], [2.0 * offset]], dtype=jnp.float32)

    def compute(x):
      return exploration.episodic_memory_intrinsic_rewards(
          x, neighbors, 0.5, max_memory_size=4
      )

    reward, state = (jax.jit(compute) if jit else compute)(embeddings)
    memory = np.full((4, 1), np.inf)
    memory[:neighbors] = 0.0
    expected, total, count = _reference(
        embeddings, memory, neighbors, scale=0.5
    )
    np.testing.assert_allclose(state.distance_sum, total, rtol=1e-6)
    self.assertEqual(int(state.distance_count), count)
    np.testing.assert_allclose(reward, expected, rtol=1e-5)
    np.testing.assert_array_equal(state.memory[:2], embeddings)

  @parameterized.parameters(False, True)
  def test_running_statistics_and_rewards_across_ring_updates(self, jit):
    memory = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [0.3, 0.1]])
    total, count, next_index = 0.15, 5, 3
    state = exploration.IntrinsicRewardState(
        memory=jnp.asarray(memory),
        distance_sum=total,
        distance_count=count,
        next_memory_index=next_index,
    )

    def compute(x, previous):
      return exploration.episodic_memory_intrinsic_rewards(
          x, 2, 1.0, previous, max_memory_size=4
      )

    compute = jax.jit(compute) if jit else compute
    for data in ([[0.01, 0.0], [0.0, 0.05]], [[0.03, 0.07], [0.25, 0.05]]):
      embeddings = jnp.array(data, dtype=jnp.float32)
      expected, total, count = _reference(embeddings, memory, 2, total, count)
      reward, state = compute(embeddings, state)
      np.testing.assert_allclose(reward, expected, rtol=1e-5)
      np.testing.assert_allclose(state.distance_sum, total, rtol=1e-6)
      self.assertEqual(int(state.distance_count), count)
      indices = (np.arange(len(data)) + next_index) % len(memory)
      memory[indices] = np.asarray(embeddings)
      next_index = (next_index % len(memory)) + len(data)
      np.testing.assert_allclose(state.memory, memory)

  @parameterized.parameters(False, True)
  def test_identical_embeddings_keep_zero_distance_statistics(self, jit):
    def compute(x):
      return exploration.episodic_memory_intrinsic_rewards(
          x, 2, 1.0, max_memory_size=4
      )

    reward, state = (jax.jit(compute) if jit else compute)(jnp.zeros((2, 1)))
    self.assertEqual(float(state.distance_sum), 0.0)
    np.testing.assert_allclose(reward, 1.0 / (np.sqrt(2.0) + 1e-3), rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
