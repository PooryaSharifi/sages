import tensorflow as tf
import numpy as np
import time
import random
import gym
from cards import call_tarot, play, Tarot

rates = [0.2, 0.1, 0.05, 0.01, 0.01, 0.01] + [.005] * 8 + [.002] * 16 + [.001] * 32 + [.0005] * 64

# / ---------------- \
#     now the gen
# \ ---------------- /


def model(n, m):
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(n * 3 / 2, input_shape=(n, ), use_bias=False, kernel_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n * 3 / 2, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activation='relu'),
        tf.keras.layers.Dense(m, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activation='softmax'),
    ])


env = gym.make('CartPole-v0')
env._max_episode_steps = 10000
env.reset()

population_size = 128
population = [model(52 * 3 * 2, 52) for _ in range(population_size)]
play(population[: 4], story_telling=True)
max_generation = 100


def mature_and_replace(strong, weak, magnitude):
    weak.set_weights(strong.get_weights())
    for layer in weak.layers:
        weights = [w + tf.random.normal(w.shape, 0, magnitude) for w in layer.get_weights()]
        layer.set_weights(weights)


for gen in range(max_generation):
    _n = time.time()
    scores = [0] * population_size

    # # run turnoment and set score
    # for genome in population:
    #     state, done = tf.random.normal((1, 4)), False
    #     steps = 0
    #     while not done:
    #         action = 0 if genome(state, training=False) < .5 else 1
    #         state, reward, done, info = env.step(action)
    #         state = tf.constant(state.reshape(1, 4).astype('float32'))
    #         steps += 1
    #     env.reset()
    #     scores.append(steps - abs(state[0][0]) * 80 - abs(state[0][2]) * 4)

    # run turnoment and set score
    for _ in range(7):
        turns = list(range(population_size))
        random.shuffle(turns)
        turns = [(
            (population[turns[i]], population[turns[(i + 1) % 4]], population[turns[(i + 2) % 4]], population[turns[(i + 3) % 4]]),
            (turns[i], turns[(i + 1) % 4], turns[(i + 2) % 4], turns[(i + 3) % 4])
        ) for i, _ in enumerate(turns)]
        for turn, indexes in turns:
            winners = play(turn)
            scores[indexes[0 + winners]] += 1
            scores[indexes[2 + winners]] += 1

    # sort
    population, scores = zip(*sorted([*zip(population, scores)], key=lambda value: -value[1]))
    population, scores = list(population), list(scores)

    # mature
    head = 0
    tail = len(population) - 1
    while len(population) * 2 < tail * 3:
        mature_and_replace(population[head], population[tail], rates[gen] * ((scores[0] - scores[tail]) / 100) ** .75)
        tail -= 1
        head += 1 if random.random() < .9 else 0
    population[tail] = model(52 * 3 * 2, 52)
    population[tail - 1] = model(52 * 3 * 2, 52)
    print(f'generation: {gen}; best score: {scores[0]}; took time: {time.time() - _n}')
    population[0].save_weights('Court_piece.h5')
    play(population[: 4], story_telling=True)


for _ in range(10):
    state, done = tf.random.normal((1, 4)), False
    steps = 0
    while not done and steps < 100000:
        action = 0 if population[0](state, training=False) < .5 else 1
        state, reward, done, info = env.step(action)
        state = tf.constant(state.reshape(1, 4).astype('float32'))
        steps += 1
        env.render()
    env.reset()
