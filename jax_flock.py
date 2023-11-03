from flax import struct
import functools
import jax
import jax.lax as lax
import jax.numpy as jp
import math

from jax.typing import ArrayLike


@struct.dataclass
class Config:
    boid_count: int
    sphere_radius: float

    mass: float = 1.0
    body_radius: float = 0.5
    max_force: float = 0.6
    max_speed: float = 0.3
    refresh_rate: float = 0.5

    exponent_separate: int = 2
    exponent_align: int = 3
    exponent_cohere: int = 1

    max_dist_separate: float = 4.0
    max_dist_align: float = 6.0
    max_dist_cohere: float = 100.0


@struct.dataclass
class Params:
    weight_forward: float
    weight_separate: float
    weight_align: float
    weight_cohere: float
    weight_avoid: float


@struct.dataclass
class State:
    position: jax.Array
    forward: jax.Array
    speed: jax.Array
    steer_memory: jax.Array
    up_memory: jax.Array
    # neighbors: jax.Array
    # time_since_refresh: jax.Array
    # TODO(mshang): cache


def blend(v_old: ArrayLike, v_new: ArrayLike, old_weight: float) -> ArrayLike:
    return v_old * old_weight + v_new * (1 - old_weight)


def unit_sigmoid(x: ArrayLike) -> ArrayLike:

    def logistic(x, k, L, x0):
        x = jp.maximum(x, -50)
        return L / (1 + jp.exp(-k * (x - x0)))

    return logistic(x, 12, 1, 0.5)


def length(v: jax.Array) -> float:
    return jp.linalg.norm(v)


def lengths(vs: jax.Array, keepdims=True) -> jax.Array:
    return jp.linalg.norm(vs, axis=1, keepdims=keepdims)


def normalize(v: jax.Array) -> jax.Array:
    return v / length(v)


def normalize_or_zero(v: jax.Array) -> jax.Array:
    leng = length(v)
    return lax.cond(leng > 0, lambda: v / leng, lambda: v)


def random_unit_vector(key: jax.random.KeyArray) -> jax.Array:
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(key, (1, )) * 2 * math.pi
    z = jax.random.uniform(subkey, (1, )) * 2 - 1
    r = jp.sqrt(1 - z**2)
    return jp.hstack([r * jp.cos(theta), r * jp.sin(theta), z])


def init_boid(config: Config, key: jax.random.KeyArray) -> State:
    key, subkey = jax.random.split(key)
    position = random_unit_vector(key) * config.sphere_radius / 2
    forward = random_unit_vector(subkey)
    speed = config.max_speed * 0.6

    return State(position=position,
                 forward=forward,
                 speed=speed,
                 steer_memory=jp.zeros_like(position),
                 up_memory=jp.zeros_like(position))


def init_state(config: Config, key: jax.random.KeyArray) -> State:
    init_boids = jax.vmap(functools.partial(init_boid, config))
    return init_boids(jax.random.split(key, config.boid_count))


def nearest_neighbors(position: jax.Array,
                      all_positions: jax.Array,
                      k=7) -> jax.Array:
    dists = lengths(position - all_positions, keepdims=False)
    indices = lax.top_k(-dists, k + 1)[1]
    return indices[1:]


def steer_to_separate(position: jax.Array, neighbors: jax.Array,
                      config: Config) -> jax.Array:

    def separation_force(neighbor: jax.Array) -> jax.Array:
        offset = position - neighbor
        dist = length(offset)
        weight = 1 / (dist**config.exponent_separate)
        weight = weight * (1 - unit_sigmoid(dist / config.max_dist_separate))
        return offset * weight

    direction = jp.sum(jax.vmap(separation_force)(neighbors), axis=0)
    return normalize_or_zero(direction)


def steer_to_align(position: jax.Array, forward: jax.Array,
                   neighbor_positions: jax.Array, neighbor_forwards: jax.Array,
                   config: Config) -> jax.Array:

    def align_force(neighbor_position: jax.Array,
                    neighbor_forward: jax.Array) -> jax.Array:
        heading_offset = neighbor_forward - forward
        dist = length(neighbor_position - position)
        weight = 1 / (dist**config.exponent_align)
        weight = weight * (1 - unit_sigmoid(dist / config.max_dist_align))
        return normalize_or_zero(heading_offset) * weight

    direction = jp.sum(jax.vmap(align_force,
                                in_axes=(0, 0))(neighbor_positions,
                                                neighbor_forwards),
                       axis=0)
    return normalize_or_zero(direction)


def steer_to_cohere(position: jax.Array, neighbors: jax.Array,
                    config: Config) -> jax.Array:
    dists = lengths(position - neighbors)
    weights = 1 / (dists**config.exponent_cohere)
    weights = weights * (1 - unit_sigmoid(dists / config.max_dist_cohere))
    neighbor_center = jp.sum(neighbors * weights, axis=0) / jp.sum(weights)
    return normalize_or_zero(neighbor_center - position)


def steer_to_avoid(position: jax.Array, config: Config) -> jax.Array:
    # TODO(mshang): do collision prediction instead of force field
    leng = length(position)
    dist = config.sphere_radius - leng
    weight = jp.where(dist < config.sphere_radius * 0.1, 1, 0)
    return -position / leng * weight


def steer(state: State, time_step: float, params: Params, config: Config,
          boid: State, index: int) -> State:
    # time_since_refresh = state.time_since_refresh + time_step
    # neighbors = lax.cond(time_since_refresh > config.neighbor_refresh_rate,
    #                      lambda: boid.neighbors,
    #                      lambda: get_nearest_k(boid.position, state.position))
    neighbors = nearest_neighbors(boid.position, state.position)
    positions = state.position[neighbors]
    forwards = state.forward[neighbors]

    f = params.weight_forward * boid.forward
    s = params.weight_separate * steer_to_separate(boid.position, positions,
                                                   config)
    a = params.weight_align * steer_to_align(boid.position, boid.forward,
                                             positions, forwards, config)
    c = params.weight_cohere * steer_to_cohere(boid.position, positions,
                                               config)
    o = params.weight_avoid * steer_to_avoid(boid.position, config)

    force = f + s + a + c + o
    steering_force = blend(boid.steer_memory, force, 0.6)

    # Limit steering force
    magnitude = length(steering_force)
    limit_steering_force = jp.where(
        magnitude > config.max_force,
        steering_force / magnitude * config.max_force, steering_force)
    acceleration = limit_steering_force / config.mass

    velocity = boid.forward * boid.speed
    new_velocity = velocity + acceleration * time_step
    new_speed = length(new_velocity)
    new_forward = new_velocity / new_speed
    clipped_speed = jp.clip(new_speed, 0, config.max_speed)

    up = normalize(
        blend(boid.up_memory, acceleration + jp.array([0, 0.01, 0]), 0.999))
    new_position = boid.position + new_forward * clipped_speed

    return State(position=new_position,
                 forward=new_forward,
                 speed=clipped_speed,
                 steer_memory=steering_force,
                 up_memory=up)


def next_state(params: Params, config: Config, time_step: float,
               state: State) -> State:
    step = jax.vmap(functools.partial(steer, state, time_step, params, config),
                    in_axes=(0, 0))
    return step(state, jp.arange(config.boid_count))

def pairwise_distances(state: State) -> jax.Array:
    positions = state.position
    n_agents = positions.shape[0]
    idx = jp.triu_indices(n_agents, k=1)
    return jp.linalg.norm(positions[idx[0]] - positions[idx[1]], axis=-1)

def avg_sep(state: State) -> float:
    return jp.mean(pairwise_distances(state))

def min_sep(state: State) -> float:
    return jp.min(pairwise_distances(state))