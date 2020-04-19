import random
import numpy as np
from neat import neat
import time
import tensorflow as tf


class Tarot:
    flag = [[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    cards_name = ["♦", "♥", "♠", "♣"]

    def __init__(self, king, choose_tarot=lambda cards: random.randint(0, 3), story_telling=False):
        self.cards = list(range(1, 5)) * 13
        self.event_horizon = [0] * 13 * 4
        self.deck = [0] * 13 * 4
        self.stack = []
        self.sit = king
        self.wins = [0, 0]
        self.story_telling = story_telling

        random.shuffle(self.cards)
        self.tarot = choose_tarot([(int(i / 13), i % 13) for i, owner in enumerate(self.cards) if owner == king][: 5])
        if story_telling:
            print(f'tarot: {self.cards_name[self.tarot]}, king: {self.sit}')

    def reform(self, cards, with_owned=False):
        return sum([self.flag[0] if owner == 0 else self.flag[(owner + 4 - self.sit) % 4 + 1] for i, owner in enumerate(cards)], [])

    def move(self, pr):
        background = int(self.stack[0] / 13) if self.stack else 4
        cards = [1 if owner == self.sit and self.event_horizon[i] == 0 else 0 for i, owner in enumerate(self.cards)]
        for i, owned in enumerate(cards):
            if owned:
                if int(i / 4) == background:
                    continue
                background = 4
        cards = [owned * (float(pr[i]) + .0000001) if background == 4 or int(i / 4) == background else 0 for i, owned in enumerate(cards)]
        best_card = np.argmax(cards)
        self.stack.append(best_card)
        self.deck[best_card] = self.sit + 1
        if len(self.stack) == 4:
            if self.story_telling:
                print((self.sit + 3) % 4 + 1, *[f'{(c % 13) + 1} {self.cards_name[int(c / 13)]} ' for c in self.stack])
            self.sit = (np.argmax([card % 13 if int(self.stack[0] / 13) == int(card / 13) else 0 + (52 if int(card / 13) == self.tarot else 0) for card in self.stack]) + self.sit) % 4
            self.wins[self.sit % 2] += 1
            self.stack.clear()
            self.event_horizon = [owner if owner else self.deck[i] for i, owner in enumerate(self.event_horizon)]
            self.deck = [0] * 13 * 4
        else:
            self.sit = (self.sit + 1) % 4


def call_tarot(turn):
    species = [0] * 4
    for card in turn:
        species[card[0]] += card[1] ** 1
    return np.argmax(species)


def play(genomes, story_telling=False):
    t = Tarot(random.randint(0, 3), choose_tarot=call_tarot, story_telling=story_telling)
    while max(t.wins) < 7:
        for _ in range(4):
            x = tf.constant([t.reform(t.event_horizon, with_owned=True) + t.reform(t.deck)], dtype=tf.float32)
            pr = genomes[t.sit](x)[0]
            t.move(pr)

    return np.argmax(t.wins)


if __name__ == '__main__':
    def throw(event_horizon, deck):
        return [random.random() for _ in range(52)]

    n = time.time()
    for _ in range(1000):
        play([throw] * 4)
    print(time.time() - n)
