import tensorflow as tf
import numpy as np
from multiprocessing import Process, Array, Value
from _mario.Notebook_04_GAN_mario import a_discriminator, a_generator
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env.reset()
compressor = tf.keras.models.Model(inputs=a_discriminator.input, outputs=a_discriminator.layers[-4].output)


encode_status = {
    'small': tf.constant([[1., .0, .0]]),
    'tall': tf.constant([[.0, 1., .0]]),
    'fireball': tf.constant([[.0, .0, 1.]]),
}


def net(name):
    pos_input = tf.keras.layers.Input((2,))

    status_input = tf.keras.layers.Input((3,))

    image_input = tf.keras.layers.Input((15, 16, 128))
    hidden_image = tf.keras.layers.Conv2D(3, (3, 3), strides=(3, 3), padding='same', use_bias=False)(image_input)  # maybe no strides
    hidden_image = tf.keras.layers.Flatten()(hidden_image)

    combined = tf.keras.layers.concatenate([hidden_image, status_input, pos_input])
    combined = tf.keras.layers.Dense(12, activation='softmax')(combined)
    return tf.keras.models.Model(inputs=[image_input, status_input, pos_input], outputs=combined, name=str(name))


def call(genome, i, fitnesses, v, v_sum):
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
    # env.close()
    fitnesses[i] = info['x_pos']
    v_sum.value += info['x_pos'] / steps


class NeuroEvolution:
    def __init__(self, population_size, mutation_rate, time_machine):
        self.population = [net(i) for i in range(population_size)]
        self.mutation_rate = mutation_rate
        self.time_machine = time_machine

        self.v = 1

    def mutate(self, individual, scale=1.0):
        for layer in individual.layers:
            weights = [w + np.random.normal(loc=0, scale=scale, size=w.shape) * np.random.binomial(1, p=self.mutation_rate, size=w.shape) for w in layer.get_weights()]
            layer.set_weights(weights)
        return individual

    def crossover(self, parent1, parent2):
        child1 = net((int(parent1.name) + 1) * 10)
        child1.set_weights(parent1.get_weights())
        child2 = net((int(parent1.name) + 1) * 10)
        child2.set_weights(parent2.get_weights())
        for i, J in enumerate(parent1.layers):
            ws1 = []
            ws2 = []
            for j, _ in enumerate(J.get_weights()):
                ws1.append(child1.layers[i].get_weights()[j])
                ws2.append(child2.layers[i].get_weights()[j])
                if len(ws1[-1].shape) != 2:
                    continue
                n_neurons = ws1[-1].shape[1]
                cutoff = np.random.randint(0, n_neurons)
                ws1[-1][:, cutoff:] = parent2.layers[i].get_weights()[j][:, cutoff:].copy()
                ws2[-1][:, cutoff:] = parent1.layers[i].get_weights()[j][:, cutoff:].copy()
            child1.layers[i].set_weights(ws1)
            child2.layers[i].set_weights(ws2)
        return child1, child2

    def calculate_fitness(self):
        fitnesses = Array('d', [.0] * len(self.population))
        v_sum = Value('d', 0)
        for i, genome in enumerate(self.population):
            self.time_machine(genome, i, fitnesses, self.v, v_sum)
        # pool = [Process(target=self.time_machine.call, args=(genome, i, fitnesses, self.v, v_sum)) for i, genome in enumerate(self.population)]
        # for p in pool:
        #     p.start()
        # for p in pool:
        #     p.join()
        for i, genome in enumerate(self.population):
            genome.fitness = fitnesses[i]
            print(genome.fitness)
        self.v = v_sum.value / len(self.population) * 2 / 3

    def evolve(self, generations=20, checkpoint=1):
        n_winners = int(len(self.population) * 0.4)
        n_parents = len(self.population) - n_winners
        for epoch in range(generations):
            self.calculate_fitness()
            fitnesses = [i.fitness for i in self.population]
            sort_fitness = np.argsort(fitnesses)[::-1]
            self.population = [self.population[i] for i in sort_fitness]
            fittest_individual = self.population[0]
            if (epoch + 1) % checkpoint == 0:
                print('epoch %d, fittest individual %d with accuracy %f' % (epoch + 1, sort_fitness[0],
                                                                            fittest_individual.fitness))
            next_population = [self.population[i] for i in range(n_winners)]
            total_fitness = np.sum([np.abs(i.fitness) for i in self.population])
            parent_probabilities = [np.abs(i.fitness / total_fitness) for i in self.population]
            parents = np.random.choice(self.population, size=n_parents, p=parent_probabilities, replace=False)
            for i in np.arange(0, len(parents), 2):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                next_population += [self.mutate(child1), self.mutate(child2)]
            self.population = next_population
        return fittest_individual


generation = NeuroEvolution(60, mutation_rate=.1, time_machine=call)
best_genome = generation.evolve(generations=64)
best_genome.save_weights()
