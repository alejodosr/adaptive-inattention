#!/usr/bin/python3
"""Script for evaluation of trained model on Imagenet VID 2015 dataset.
Few global variables defined here are explained:
Global Variables
----------------
args : dict
	Has all the options for changing various variables of the model as well as parameters for evaluation
dataset : ImagenetDataset (torch.utils.data.Dataset, For more info see datasets/vid_dataset.py)

"""
from utils.misc import str2bool
import argparse
import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines.common.callbacks import BaseCallback

from rl.rl_agent.custom_policies import CropPolicy, BaselinePolicy, CropPolicyYOTO
from rl.rl_env.inatt_env import HrmodEnv


from datetime import datetime
import os

parser = argparse.ArgumentParser(description="MVOD Evaluation on VID dataset")
parser.add_argument('--net', default="lstm5",
                    help="The network architecture, it should be of backbone, lstm.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--tags_csv", type=str)
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--dataset_type", default="imagenet_vid", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument('--gpu_id', default=0, type=int,
                    help='The GPU id to be used')
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--lambda_0", type=float, default=2.0, help="Lambda_0 for reward function.")
parser.add_argument("--eval_dir", default="temp", type=str, help="The directory to store evaluation results.")
parser.add_argument('--width_mult', default=1.0, type=float,
                    help='Width Multiplifier for network')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--train_total_steps', default=10e5, type=int,
                    help='Total training steps for the training agent')
parser.add_argument('--train_steps', default=20e3, type=int,
                    help='Training steps between testing')
parser.add_argument('--test_steps', default=1e3, type=int,
                    help='Testing steps between training')
parser.add_argument("--normalize_env", type=str2bool, default=True)
parser.add_argument('--stop_norm', default=58e3, type=int,
                    help='Stops observation normalization after requested training steps')
parser.add_argument("--dynamic_gamma", type=str2bool, default=True)
parser.add_argument('--it', default=4, type=int,
                    help='Training steps between testing')

args = parser.parse_args()

if __name__ == '__main__':

    # Experiment dir
    results_path = "checkpoints/rl_logs/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p") + "_train"
    checkpoint_path = results_path + '/checkpoints'
    if args.resume is not None:
        results_path = args.resume
        checkpoint_path = os.path.join(args.resume, "checkpoints")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create and wrap the environment
    h_shape = (10, 10, 1024)  # Shape of the hidden state of the lstm network
    history_shape = 20  # Number of past actions to be tracked
    env = HrmodEnv(args, h_shape=h_shape, history_shape=history_shape, val_dataset=True, random_train=True, cache_dir=checkpoint_path, dynamic_gamma=args.dynamic_gamma)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    if args.normalize_env:
        print("RL_TRAIN: Normalizing input observation features")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        env.training = True

    # Instantiate agent
    cliprange = 0.1
    nminibatches = 4
    if args.dynamic_gamma:
        agent = PPO2(CropPolicyYOTO, env, verbose=1, tensorboard_log=results_path + "/tb", cliprange=cliprange, nminibatches=nminibatches)
        print("RL_TRAIN: Dynamic gamma activated")
    else:
        agent = PPO2(CropPolicy, env, verbose=1, tensorboard_log=results_path + "/tb", cliprange=cliprange, nminibatches=nminibatches)

    # Load if pretrained
    if args.resume is not None:
        del agent
        agent = PPO2.load(checkpoint_path + "/best_agent.zip", env=env, verbose=1, tensorboard_log=results_path + "/tb",
                          cliprange=cliprange, nminibatches=nminibatches)
        print("INFO: Loaded model " + args.resume)

        if args.normalize_env:
            env = VecNormalize.load(checkpoint_path + '/vecnormalize.pkl', env)

    # Variables
    best_reward = -20
    best_pct_cropped_a = -20
    best_map_diff = -20
    best_rl_map = -20
    tb_text = 'No results yet'

    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            self.is_tb_set = False
            self.meta = tf.SummaryMetadata()
            self.meta.plugin_data.plugin_name = "text"
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:
            # Log scalar value
            value = env.get_attr("diff_loss", 0)[0]
            summary = tf.Summary(value=[tf.Summary.Value(tag='differences_in_losses', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = env.get_attr("nb_cropped", 0)[0].sum().item()
            summary = tf.Summary(value=[tf.Summary.Value(tag='number_cropped_in_100', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = best_reward
            summary = tf.Summary(value=[tf.Summary.Value(tag='best_test_reward', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = best_map_diff
            summary = tf.Summary(value=[tf.Summary.Value(tag='best_map_diff', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = best_rl_map
            summary = tf.Summary(value=[tf.Summary.Value(tag='best_rl_map', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = env.get_attr("gamma_r", 0)[0]
            summary = tf.Summary(value=[tf.Summary.Value(tag='train_r_gamma', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            # Log scalar value
            value = best_pct_cropped_a
            summary = tf.Summary(value=[tf.Summary.Value(tag='best_test_pct_cropped', simple_value=value)])
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            value = tb_text
            text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
            summary = tf.Summary()
            summary.value.add(tag="eval_results", metadata=self.meta, tensor=text_tensor)
            self.locals['writer'].add_summary(summary, self.num_timesteps)
            return True


    def evaluate(model, num_steps=1000, it=4):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :param it: (int) number of iterations of num_steps evaluation
        :return: (float) Mean reward for the last 100 episodes
        """

        if args.dynamic_gamma:
            gammas = [0.01, 0.5, 1.5, 1.9]
            assert len(gammas) == it

        episode_rewards = [0.0]
        obs = env.reset()
        env.training = False    # Do not compute stats for obs normalization
        env.env_method('set_is_eval', True)
        accum_map_rl = 0
        cropped_probs = []

        # Current iteration info
        results_text = """*** Last evaluation results ***"""
        results_text = results_text + "  \n-- RL agent --"
        info_it = {'result_text': results_text, 'pct_cropped': cropped_probs}

        for j in range(it):
            obs = env.reset()
            nb_cropped_actions = 0.0
            if args.dynamic_gamma:
                env.env_method('set_gamma_r', gammas[j])
            for i in range(num_steps):
                # _states are only useful when using LSTM policies
                action, _states = model.predict(obs)
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = env.step(action)

                # Stats
                episode_rewards += reward[0]
                nb_cropped_actions += action
                if done:
                    print("RL_TEST: Reseting")
                    obs = env.reset()
            # Compute mAP
            last_map = env.env_method('compute_map')[0]
            accum_map_rl += last_map
            cropped_probs.append(nb_cropped_actions[0] / num_steps)

            # Generate text
            if args.dynamic_gamma:
                results_text = results_text + "  \n  \t*Gamma:* " + str(gammas[j]) + "  \n  \tmAP: " + str(last_map) + "  \n  \tpct_cropped: " + str(nb_cropped_actions[0] * 100 / num_steps)
            else:
                results_text = results_text + "  \n  \t*Gamma:* " + str(args.gamma) + "  \n  \tmAP: " + str(last_map) + "  \n  \tpct_cropped: " + str(nb_cropped_actions[0] * 100 / num_steps)

        # Generate info
        info_it['pct_cropped'] = cropped_probs

        # Mean mAP
        mean_map_rl = accum_map_rl / it

        # Compute mean reward for the last 100 episodes
        mean_100ep_reward = episode_rewards / (num_steps * it)

        # Compute mAP for baseline
        results_text = results_text + "  \n-- Random agent --"
        baseline_policy = BaselinePolicy(0.5)
        accum_map_baseline = 0
        for j in range(it):
            env.reset()
            baseline_policy.prob = cropped_probs[j]
            baseline_nb_cropped_act = 0.0
            for i in range(num_steps):
                # _states are only useful when using LSTM policies
                action = baseline_policy.predict()
                # here, action, rewards and dones are arrays
                # because we are using vectorized env
                obs, reward, done, info = env.step([action])

                # Stats
                episode_rewards += reward[0]
                baseline_nb_cropped_act += action
                if done:
                    print("RL_TEST: Reseting")
                    obs = env.reset()
            # Compute mAP
            last_baseline_map = env.env_method('compute_map')[0]
            accum_map_baseline += last_baseline_map
            results_text = results_text + "  \n  \tmAP: " + str(last_baseline_map) + \
                           "  \n  \tpct_cropped: " + str(baseline_nb_cropped_act * 100 / num_steps)

        # Mean mAP
        mean_map_baseline = accum_map_baseline / it

        # Compute map diff
        map_diff = mean_map_rl - mean_map_baseline

        # Compute PCT of number of cropped actions
        pct_nb_actions = np.asarray(cropped_probs).sum() * 100 / it

        # Generate info
        info_it['result_text'] = results_text

        print("RL_TEST: Mean RL policy reward:", mean_100ep_reward)
        print("RL_TEST: Number of RL policy cropped actions:", pct_nb_actions)
        print("RL_TEST: Rl map: " + str(mean_map_rl))
        print("RL_TEST: Baseline map: " + str(mean_map_baseline))
        print("RL_TEST: Diff map: " + str(map_diff))

        env.training = True  # Re-enable stats for obs normalization
        env.env_method('set_is_eval', False)  # Re-enable training computation

        return mean_100ep_reward, pct_nb_actions, map_diff, mean_map_rl, info_it

    # Train the agent
    t = 0
    args.test_steps = len(env.get_attr("dataset", 0)[0])
    while t < args.train_total_steps:

        # Train model
        if t == 0:
            agent.learn(total_timesteps=int(args.train_steps), log_interval=100, callback=TensorboardCallback())
        else:
            agent.learn(total_timesteps=int(args.train_steps), log_interval=100, callback=TensorboardCallback(),
                        reset_num_timesteps=False)

        # Evaluate
        mean_r, pct_cropped_a, map_diff, rl_map, info_it = evaluate(agent, num_steps=int(args.test_steps), it=args.it)

        tb_text = info_it['result_text']

        # Check if gamma is certainly influencing behavior
        is_gamma_infl = False
        if args.dynamic_gamma:
            cropped_probs = info_it['pct_cropped']
            for k in range(args.it - 1):
                if cropped_probs[k] < cropped_probs[k+1]:
                    is_gamma_infl = True
                else:
                    is_gamma_infl = False
                    break
        else:
            is_gamma_infl = True

        # Save if best
        if map_diff > best_map_diff and is_gamma_infl:
            # Indicate that was the best
            tb_text = tb_text + '  \nThis is the current best'

            if t > args.stop_norm:
                tb_text = tb_text + '  \n(observation norm has been disabled)'

            # Remove after using it
            if os.path.exists(checkpoint_path + '/best_agent.zip'):
                os.remove(checkpoint_path + '/best_agent.zip')

            best_map_diff = map_diff
            best_rl_map = rl_map
            best_reward = mean_r
            best_pct_cropped_a = pct_cropped_a
            agent.save(checkpoint_path + '/best_agent.zip')
            print("RL_TRAIN: Saving best model in " + checkpoint_path + '/best_agent.zip')

            if args.normalize_env:
                # Remove after using it
                if os.path.exists(checkpoint_path + '/vecnormalize.pkl'):
                    os.remove(checkpoint_path + '/vecnormalize.pkl')
                # Important: save the running average, for testing the agent we need that normalization
                agent.get_vec_normalize_env().save(checkpoint_path + '/vecnormalize.pkl')

                #  Debug
                # Remove after using it
                if os.path.exists(checkpoint_path + '/vecnormalize2.pkl'):
                    os.remove(checkpoint_path + '/vecnormalize2.pkl')
                # Important: save the running average, for testing the agent we need that normalization
                env.save(checkpoint_path + '/vecnormalize2.pkl')

        if t > args.stop_norm:
            print("RL_TRAIN: Observation normalization has been disabled")
            env.training = False


        # Update t
        t = t + args.train_steps

    env.close()