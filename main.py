from __future__ import division

import argparse
import os
import time
from tqdm import tqdm

import gym
import numpy as np

from gym_utils.replay_memory_wrapper import ReplayMemoryWrapper
from gym_utils.tensorflow_agents import SimpleQNetAgent
from gym_utils.tensorflow_models import mlp

def main(args):

    # Make Environment
    env = gym.make(args.env)
    
    model_fn = mlp
    
    if args.atari == True:
        from gym_utils.image_wrappers import ImgGreyScale, ImgResize, ImgFrameHistoryWrapper
        from gym_utils.frame_skip_wrapper import FrameSkipWrapper
        from gym_utils.tensorflow_models import deepmind_CNN
        env = FrameSkipWrapper(env, 4)
        env = ImgResize(env)
        env = ImgGreyScale(env)
        env = ImgFrameHistoryWrapper(env, 4)
        model_fn = deepmind_CNN
    
    env = ReplayMemoryWrapper(env)
    memory = env.get_memory()
    
    state = env.reset()
    
    # Set up agent
    agent = SimpleQNetAgent(
        env.observation_space.shape,
        env.action_space.n,
        model_fn,
        args.discount,
        args.epsilon,
        args.learning_rate,
        args.double_q,
        args.target_step
    )
    
    ep_rewards = []
    ep_r = 0
    
    # Keep training until reach max iterations
    for step in tqdm(range(args.training_iters), ncols=70):

        # Act
        #state = episode.get_state()
        act = agent.getAction(state)
        state, reward, terminal, _ = env.step(act)

        # Keep track of total episode reward
        ep_r += reward
            
        if terminal:
            # Reset environment
            env.reset()
            ep_rewards.append(ep_r); ep_r = 0

        # Train 
        if (memory.count >= args.batch_size*2):
            # Get transition sample from memory
            s_t0, a_t0, r_t1, s_t1, t_t1 = memory.sample(args.batch_size)
            # Run optimization op (backprop)
            agent.tdUpdate(s_t0, a_t0, r_t1, s_t1, t_t1)


        # Display Statistics
        if (step) % args.display_step == 0:
             if len(ep_rewards) is not 0:
               max_ep_r = np.amax(ep_rewards); avr_ep_r = np.mean(ep_rewards)
             else: 
               max_ep_r = 0 ; avr_ep_r = 0
               
             tqdm.write("{}, {:>7}/{}it | avr_ep_r: {:4.1f}, max_ep_r: {:4.1f}, num_eps: {}"\
                        .format(time.strftime("%H:%M:%S"), 
                        step, 
                        args.training_iters, 
                        avr_ep_r, 
                        max_ep_r, 
                        len(ep_rewards)))
                        
             ep_rewards = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0',
                       help='Name of Gym environment')
    parser.add_argument('--atari', type=int, default=0,
                       help='Automatically wrap env with image wrappers')

    parser.add_argument('--training_iters', type=int, default=100000,
                       help='Number of training iterations to run for')
    parser.add_argument('--display_step', type=int, default=2500,
                       help='Number of iterations between parameter prints')

    parser.add_argument('--batch_size', type=int, default=32,
                       help='Size of batch for Q-value updates')

    parser.add_argument('--use_target', type=bool, default=True,
                       help='Use separate target network')
    parser.add_argument('--target_step', type=int, default=1000,
                       help='Steps between updates of the taget network')
    parser.add_argument('--double_q', type=int, default=1,
                       help='Use Double Q learning')

    parser.add_argument('--discount', type=float, default=0.9,
                       help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Initial epsilon')
    parser.add_argument('--learning_rate', type=float, default=0.00025,
                       help='Learning rate for TD updates')
                       
    parser.add_argument('--verbose_tf', type=int, default=0,
                       help='Display tensorflow warnings')


    args = parser.parse_args()

    arg_dict = vars(args)
    print(' ' + '_'*33 + ' ')
    print('|' + ' '*16 + '|' + ' '*16  + '|')
    for i in arg_dict:
        print "|{:>15} | {:<15}|".format(i, arg_dict[i])
    print('|' + '_'*16 + '|' + '_'*16  + '|')
    print('')
    
    if not args.verbose_tf:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    main(args)

