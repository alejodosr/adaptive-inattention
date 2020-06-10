import gym
import tensorflow as tf
import numpy as np
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

import random

class CropPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, h_shape=[23, 23, 1024], **kwargs):
        super(CropPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):

            # Get tensors
            nu_bbox = self.processed_obs[:, :, -24:]
            nu_bbox = tf.reshape(nu_bbox, [tf.shape(nu_bbox)[0], nu_bbox.shape[2]])
            h_obs = self.processed_obs[:, :, :-24]
            h_obs = tf.reshape(h_obs, [tf.shape(h_obs)[0], h_shape[0], h_shape[1], h_shape[2]])

            activ = tf.nn.relu
            layer_1 = activ(
                conv(h_obs, 'c1', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv(layer_1, 'c2', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = tf.layers.max_pooling2d(layer_2, 2, 2, name='pool1')
            layer_3 = conv_to_fc(layer_3)
            extracted_features = activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

            extracted_features = tf.layers.flatten(extracted_features)
            extracted_features = tf.concat([extracted_features, nu_bbox], axis=1)  # Concatenate history term

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CropPolicyYOTO(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, h_shape=[10, 10, 1024], **kwargs):
        super(CropPolicyYOTO, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):

            # Get tensors
            nu_bbox = self.processed_obs[:, :, -25:-1]
            # nu_bbox = tf.Print(nu_bbox, [nu_bbox[:, :, -4:]], "nu_bbox = ")
            nu_bbox = tf.reshape(nu_bbox, [tf.shape(nu_bbox)[0], nu_bbox.shape[2]])
            h_obs = self.processed_obs[:, :, :-25]
            h_obs = tf.reshape(h_obs, [tf.shape(h_obs)[0], h_shape[0], h_shape[1], h_shape[2]])

            gamma = self.processed_obs[:, :, -1:]
            gamma = tf.reshape(gamma, [tf.shape(gamma)[0], gamma.shape[2]])
            # gamma = tf.Print(gamma, [gamma], "gamma = ")

            activ = tf.nn.relu
            x = activ(
                conv(h_obs, 'c1', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            mean_mlp, std_mlp = self.create_mlp_for_yoto(gamma=gamma, out_layer_size=128, hidden_layer_size=64,
                                                         layer_name='c1', reuse=reuse)
            mean_mlp = tf.expand_dims(mean_mlp, 1)
            mean_mlp = tf.expand_dims(mean_mlp, 1)
            std_mlp = tf.expand_dims(std_mlp, 1)
            std_mlp = tf.expand_dims(std_mlp, 1)

            x = tf.multiply(std_mlp, x)
            x = tf.add(mean_mlp, x)

            x = activ(conv(x, 'c2', n_filters=128, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            mean_mlp, std_mlp = self.create_mlp_for_yoto(gamma=gamma, out_layer_size=128, hidden_layer_size=64,
                                                         layer_name='c2', reuse=reuse)

            mean_mlp = tf.expand_dims(mean_mlp, 1)
            mean_mlp = tf.expand_dims(mean_mlp, 1)
            std_mlp = tf.expand_dims(std_mlp, 1)
            std_mlp = tf.expand_dims(std_mlp, 1)

            x = tf.multiply(std_mlp, x)
            x = tf.add(mean_mlp, x)

            x = tf.layers.max_pooling2d(x, 2, 2, name='pool1')
            x = conv_to_fc(x)
            x = activ(linear(x, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
            mean_mlp, std_mlp = self.create_mlp_for_yoto(gamma=gamma, out_layer_size=512, hidden_layer_size=128,
                                                         layer_name='fc1', reuse=reuse)
            x = tf.multiply(std_mlp, x)
            x = tf.add(mean_mlp, x)
            extracted_features = x

            extracted_features = tf.layers.flatten(extracted_features)
            extracted_features = tf.concat([extracted_features, nu_bbox], axis=1)  # Concatenate history term

            pi_h = extracted_features
            for i, layer_size in enumerate([128, 128, 128]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def create_mlp_for_yoto(self, gamma, out_layer_size, hidden_layer_size, layer_name, reuse):
        with tf.variable_scope("model", reuse=reuse):
            x = tf.nn.relu(tf.layers.dense(gamma, hidden_layer_size, name=layer_name + '_mean_1'))
            mean_mlp = tf.nn.relu(tf.layers.dense(x, out_layer_size, name=layer_name + '_mean_out'))

            x = tf.nn.relu(tf.layers.dense(gamma, hidden_layer_size, name=layer_name + '_std_1'))
            std_mlp = tf.nn.relu(tf.layers.dense(x, out_layer_size, name=layer_name + '_std_out'))

            return mean_mlp, std_mlp


class BaselinePolicy:
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, prob=0.5):
        self.prob = prob
        # self.counter = 0

    def predict(self):
        if random.uniform(0.0, 1.0) < self.prob:
            action = 1
        else:
            action = 0

        return action

    def compute_observation(self, model=None, last_action=None):
        return 0