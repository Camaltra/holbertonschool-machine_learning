from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariProcessor(Processor):
    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


def build_model(num_action):
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_action))
    model.add(Activation('linear'))

    return model


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_action = env.action_space.n
    window = 4
    model = build_model(num_action)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=num_action, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=1750000,
            log_interval=10000,
            visualize=False,
            verbose=1)
    dqn.save_weights('policy.h5', overwrite=True)
