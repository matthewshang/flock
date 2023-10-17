from flax import struct
import functools
import jax
import jax.lax as lax
import jax.numpy as jnp
import math


@struct.dataclass
class StaticParams:
    boid_count: int
    sphere_radius: float
    max_force = 0.6
    max_speed = 0.3
    exponent_separate = 2
    exponent_align = 3
    exponent_cohere = 1


@struct.dataclass
class OptimizedParams:
    weight_forward: float
    weight_separate: float
    weight_align: float
    weight_cohere: float
    weight_avoid: float


@struct.dataclass
class State:
    positions: jnp.ndarray
    forwards: jnp.ndarray
    speeds: jnp.ndarray


def normalize_or_zero(v: jnp.ndarray) -> jnp.ndarray:
    length = jnp.linalg.norm(v)
    return lax.cond(length > 0, lambda: v / length, lambda: v)


def index_mask(i: int, n: int) -> jnp.ndarray:
    """Returns a mask of length n with all True values except for index i.
    Used to avoid normalizing the vector [x_i - x_i = 0], which results in NaN
    from trying to evaluate the gradient of [sqrt] at 0."""
    return jnp.arange(n - 1) + (jnp.arange(n - 1) >= i)


def random_unit_vector(key: jnp.ndarray) -> jnp.ndarray:
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(key, (1, )) * 2 * math.pi
    z = jax.random.uniform(subkey, (1, )) * 2 - 1
    r = jnp.sqrt(1 - z**2)
    return jnp.hstack([r * jnp.cos(theta), r * jnp.sin(theta), z])


def init_state(seed: int, cfg: StaticParams) -> State:
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    make_dirs = jax.vmap(random_unit_vector)
    positions = make_dirs(jax.random.split(
        key, cfg.boid_count)) * cfg.sphere_radius / 2
    forwards = make_dirs(jax.random.split(subkey, cfg.boid_count))
    speeds = cfg.max_speed * 0.6 * jnp.ones((cfg.boid_count, 1))

    return State(positions=positions, forwards=forwards, speeds=speeds)


def get_nearest_k(positions: jnp.ndarray) -> jnp.ndarray:

    def nearest_k(me: jnp.ndarray, i: int, others: jnp.ndarray):
        # TODO(mshang): ignore self
        dists = jnp.linalg.norm(me - others, axis=1)
        _values, indices = lax.top_k(-dists, k=7)
        return indices

    return jax.vmap(nearest_k, in_axes=(0, None), out_axes=0)(positions,
                                                              positions)


# Version of [steer_to_separate] using positions of nearest neighbors.
def steer_to_separate_nn(positions: jnp.ndarray,
                         nearest_positions: jnp.ndarray,
                         cfg: StaticParams) -> jnp.ndarray:

    def force_one_neighbor(me: jnp.ndarray,
                           neighbor: jnp.ndarray) -> jnp.ndarray:
        offset = me - neighbor
        dist = jnp.linalg.norm(offset)
        weight = 1 / (dist**cfg.exponent_separate)
        return offset * weight

    def force_all_neighbors(me: jnp.ndarray,
                            neighbors: jnp.ndarray) -> jnp.ndarray:
        direction = jnp.sum(jax.vmap(force_one_neighbor,
                                     in_axes=(None, 0),
                                     out_axes=0)(me, neighbors),
                            axis=0)
        return normalize_or_zero(direction)

    return jax.vmap(force_all_neighbors, in_axes=(0, 0),
                    out_axes=0)(positions, nearest_positions)


def steer_to_separate(positions: jnp.ndarray,
                      cfg: StaticParams) -> jnp.ndarray:

    def force_one_neighbor(me: jnp.ndarray,
                           neighbor: jnp.ndarray) -> jnp.ndarray:
        offset = me - neighbor
        dist = jnp.linalg.norm(offset)
        weight = 1 / (dist**cfg.exponent_separate)
        return offset * weight

    def force_all_neighbors(me: jnp.ndarray, i: int) -> jnp.ndarray:
        indices = index_mask(i, positions.shape[0])
        collect_neighbors = jax.vmap(force_one_neighbor,
                                     in_axes=(None, 0),
                                     out_axes=0)
        direction = jnp.sum(collect_neighbors(me, positions[indices]), axis=0)
        return normalize_or_zero(direction)

    return jax.vmap(force_all_neighbors, in_axes=(0, 0),
                    out_axes=0)(positions, jnp.arange(positions.shape[0]))


def steer_to_align(positions: jnp.ndarray, forwards: jnp.ndarray,
                   cfg: StaticParams) -> jnp.ndarray:

    def force(my_position: jnp.ndarray, their_position: jnp.ndarray,
              my_forward: jnp.ndarray,
              their_forward: jnp.ndarray) -> jnp.ndarray:
        heading_offset = their_forward - my_forward
        dist = jnp.linalg.norm(my_position - their_position)
        weight = 1 / (dist**cfg.exponent_align)
        return normalize_or_zero(heading_offset) * weight

    def force_all_neighbors(my_position: jnp.ndarray, my_forward: jnp.ndarray,
                            i: int) -> jnp.ndarray:
        indices = index_mask(i, positions.shape[0])
        collect_neighbors = jax.vmap(force, in_axes=(None, 0, None, 0))
        direction = jnp.sum(collect_neighbors(my_position, positions[indices],
                                              my_forward, forwards[indices]),
                            axis=0)
        return normalize_or_zero(direction)

    collect_all = jax.vmap(force_all_neighbors, in_axes=(0, 0, 0))
    return collect_all(positions, forwards, jnp.arange(positions.shape[0]))


def steer_to_cohere(positions: jnp.ndarray, cfg: StaticParams) -> jnp.ndarray:

    def force(me: jnp.ndarray, i: int) -> jnp.ndarray:
        indices = index_mask(i, positions.shape[0])
        other_positions = positions[indices]
        dists = jnp.linalg.norm(me - other_positions, axis=1, keepdims=True)
        weights = 1 / (dists**cfg.exponent_cohere)
        center = jnp.sum(other_positions * weights, axis=0) / jnp.sum(weights)
        return normalize_or_zero(center - me)

    return jax.vmap(force, in_axes=(0, 0))(positions,
                                           jnp.arange(positions.shape[0]))


def steer_to_avoid(positions: jnp.ndarray, cfg: StaticParams) -> jnp.ndarray:
    # TODO(mshang): do collision prediction instead of force field

    def force(me: jnp.ndarray) -> jnp.ndarray:
        length = jnp.linalg.norm(me)
        dist = cfg.sphere_radius - length
        weight = jnp.where(dist < cfg.sphere_radius / 10, 1, 0)
        return -me / length * weight

    return jax.vmap(force)(positions)


@functools.partial(jax.jit, static_argnames=['cfg'])
def next_state(state: State, params: OptimizedParams,
               cfg: StaticParams) -> State:
    dt = 1 / 60

    positions = state.positions
    forwards = state.forwards
    speeds = state.speeds

    # nearests = get_nearest_k(positions)
    # nearest_positions = jnp.take(positions, nearests, axis=0)

    f = params.weight_forward * forwards
    # s = params.weight_separate * \
    #     steer_to_separate_nn(positions, nearest_positions)
    s = params.weight_separate * steer_to_separate(positions, cfg)
    a = params.weight_align * steer_to_align(positions, forwards, cfg)
    c = params.weight_cohere * steer_to_cohere(positions, cfg)
    o = params.weight_avoid * steer_to_avoid(positions, cfg)

    steering_force = f + s + a + c + o

    magnitude = jnp.linalg.norm(steering_force, axis=1, keepdims=True)
    steering_force = jnp.where(magnitude > cfg.max_force,
                               steering_force / magnitude * cfg.max_force,
                               steering_force)

    new_velocity = forwards * speeds + steering_force * dt
    new_speeds = jnp.linalg.norm(new_velocity, axis=1, keepdims=True)
    new_forwards = new_velocity / new_speeds
    new_speeds = jnp.clip(new_speeds, 0, cfg.max_speed)

    # TODO(mshang): why doesn't this need a dt?
    new_positions = positions + new_forwards * new_speeds

    return State(positions=new_positions,
                 forwards=new_forwards,
                 speeds=new_speeds)
