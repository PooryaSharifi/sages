import tensorflow as tf
import cv2
import numpy as np
import random
import time
from _mario.Notebook_04_GAN_mario import a_discriminator, a_generator
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()

rates = [0.2, 0.1, 0.05, 0.01, 0.01, 0.01] + [.005] * 8 + [.002] * 16 + [.001] * 32 + [.0005] * 64
compressor = tf.keras.models.Model(inputs=a_discriminator.input, outputs=a_discriminator.layers[-4].output)


def model():
    pos_input = tf.keras.layers.Input((2,))

    status_input = tf.keras.layers.Input((3,))

    image_input = tf.keras.layers.Input((15, 16, 128))
    hidden_image = tf.keras.layers.Conv2D(2, (3, 3), strides=(3, 3), padding='same', use_bias=False)(
        image_input)  # maybe no strides
    hidden_image = tf.keras.layers.Flatten()(hidden_image)

    combined = tf.keras.layers.concatenate([hidden_image, status_input, pos_input])
    combined = tf.keras.layers.Dense(12, activation='softmax')(combined)
    return tf.keras.models.Model(inputs=[image_input, status_input, pos_input], outputs=combined)


population_size = 64
population = [model() for _ in range(population_size)]

try:
    for i in range(population_size):
        population[i].load_weights(f'mario_genomes/{i}.h5')
except: pass
max_generation = 100


def mature_and_replace(strong, weak, magnitude):
    weak.set_weights(strong.get_weights())
    for layer in weak.layers:
        weights = [w + tf.random.normal(w.shape, 0, magnitude) for w in layer.get_weights()]
        layer.set_weights(weights)


encode_status = {
    'small': tf.constant([[1., .0, .0]]),
    'tall': tf.constant([[.0, 1., .0]]),
    'fireball': tf.constant([[.0, .0, 1.]]),
}

v = 1.

for gen in range(max_generation):
    _n = time.time()
    scores = []
    v_sum = 0

    # run turnoment and set score
    for genome in population:
        done, steps, action = False, 0, 3
        while not done:
            obs, reward, done, info = env.step(action)
            if steps % 2 == 0:
                obs = tf.image.rgb_to_grayscale(obs)
                small = tf.image.resize(obs, (120, 128))
                small = tf.expand_dims(small, 0)
                definitions = compressor(small, training=False)
            action = genome([definitions, encode_status[info['status']],
                            tf.constant([(info['x_pos_screen'], info['y_pos'])], dtype=tf.float32)], training=False)[0]

            action = np.argmax(action)
            steps += 1
            if info['x_pos'] / steps < v:
                done = True

            env.render()

        env.reset()
        scores.append(info['x_pos'])
        v_sum += info['x_pos'] / steps
    v = v_sum / population_size * 2 / 3

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
    population[tail] = model()
    population[tail - 1] = model()
    print(f'generation: {gen}; best score: {scores[0]}; took time: {time.time() - _n}; v updated to {v}')
    for i in range(population_size):
        population[i].save_weights(f'mario_genomes/{i}.h5')
    # play(population[: 4], story_telling=True)

