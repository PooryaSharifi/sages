# Mario
download a "mario_all_episodes.mp4" and chup it into images (120x128 resolution, 1 image per 3 frames). then train gan in Notebook_04_GAN_mario.py.
here you can see generated images from trained generator:

![Alt Text](https://www.dropbox.com/s/owjy8kc7l0vv5f9/dcgan.gif?raw=1)

now the last convolutional layer of descriminator can identify images. and have the resolution less than 120x128. it generates input for neuroevolutions in Notebook_05_evolutionary_mario.py and Notebook_06_parallelized_evolutionary_net.py to play the game.

in 54th epoch mario learn to kill enemies and jump over tall obstacles.
