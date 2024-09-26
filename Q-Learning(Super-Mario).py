import copy
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import logging
import multiprocessing
from nes_py.wrappers import JoypadSpace
import random
import time
import warnings

def frames_to_video(frames, output_video, fps=60):
    vid_length = len(frames)
    print(f'Making video with {vid_length} frames.')
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    # Get the height and width of the frames from the first frame
    height, width, layers = frames[0].shape
    size = (width, height)

    # Define the codec and create a VideoWriter object (XVID is a common codec)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Write each frame to the video file
    for frame in frames:
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video saved as {output_video}")

def trial(instruction_set, render=False, fps=60, random=False, video_name=None):
    #print(f'Running trial on: {instruction_set}')
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    frame = env.reset()

    running_reward = 0
    frames = [copy.deepcopy(frame)]
    for action in instruction_set:
        #start_time = time.time()
        #frame_time = 1.0 / fps  # Calculate the time per frame based on FPS
        # env.render(); Adding this line would show the attempts
        #if(render):
        #    env.render()
        # of the agent in a pop up window.

        # Take random action
        if(random):
            action = env.action_space.sample()

        # Apply the sampled action in our environment
        #print(action)
        frame, reward, done, info = env.step(action)
        frames.append(copy.deepcopy(frame))
        running_reward += reward

        if(done or info['life'] < 2):
            running_reward = 0
            break

    # Calculate how long to wait to maintain the desired FPS
        #elapsed_time = time.time() - start_time
        #time_to_wait = frame_time - elapsed_time
        #if time_to_wait > 0:
        #    time.sleep(time_to_wait)

    #print(f'Running reward is {running_reward}.')
    if(render):
        frames_to_video(frames, video_name, fps=fps)
    return running_reward / sum(instruction_set)

def solve(generations, individuals=240, keep=24, max_parallel=8, render_every=10, ext_per_gen=1):
    gen = recombine([[3, 3, 3], [4, 4, 4]], count=individuals)
    for i in range(generations):
        gen = remove_duplicates(gen)
        gen_count = len(gen)
        gen = run_generation(gen, keep=keep, max_parallel=max_parallel)
        best = gen[0]
        best_score = score_individual(best)
        offspring = gen + recombine(gen, individuals-keep)

        if(i % render_every == 0):
            score_individual(best, render=True, video_name=f'best_of_gen_{i}.mp4')

        for _ in range(ext_per_gen):
            offspring = extend(offspring)

        gen = gen + offspring

        print(f'Generation: {i} ({gen_count}), Max fitness this generation: {best_score}, Action Sequence: {best}')
        #time.sleep(1)

def extend(individuals):
    new_individuals = []
    for i in individuals:
        i.append(random.choice([3, 4]))
        new_individuals.append(i)
    return new_individuals

import random

def recombine(individuals, count):
    # Initialize the offspring DNA list
    offspring = []
    
    # Ensure there are at least two parents to choose from
    if len(individuals) < 2:
        raise ValueError("At least two individuals are required for recombination.")
    
    # Recombine DNA to generate `count` offspring
    for _ in range(count):
        # Randomly select two different parents from the individuals list
        parent1, parent2 = random.sample(individuals, 2)
        
        # Get the maximum genome length between the two parents
        max_length = max(len(parent1), len(parent2))
        
        # Generate a new DNA sequence by combining genes from both parents
        child = []
        for i in range(max_length):
            gene1 = parent1[i] if i < len(parent1) else None
            gene2 = parent2[i] if i < len(parent2) else None
            
            # Choose gene from one parent if the other doesn't have a gene in this position
            if gene1 is None:
                child.append(gene2)
            elif gene2 is None:
                child.append(gene1)
            else:
                child.append(random.choice([gene1, gene2]))
        
        offspring.append(child)

    offspring = [mutate(x) for x in offspring]
    
    return offspring

# Define score_individual at the top level
def score_individual(i, repeat_action=15, render=False, video_name=None):
    # Create repeated list and calculate score for an individual
    repeated_list = [item for item in i for _ in range(repeat_action)]
    return trial(repeated_list, render=render, video_name=video_name)

def run_generation(individuals, keep, max_parallel):
    # Create a pool of workers
    with multiprocessing.Pool(processes=max_parallel) as pool:
    # Map the score_individual function to individuals in parallel
        scores = pool.map(score_individual, individuals)
        scored = zip(scores, individuals)
    
    # Sort the scored individuals by their score and keep the top ones
    return [x[1] for x in sorted(scored, key=lambda x: x[0], reverse=True)][0:keep]

def mutate(sequence, mutation_rate=0.9, mutation_decay_rate=0.5):
    sequence.reverse()
    for i in range(len(sequence)):
        if(random.random() < mutation_rate):
            sequence[i] = random.choice([3, 4])
        mutation_rate = mutation_rate * mutation_decay_rate
    sequence.reverse()
    return sequence

def remove_duplicates(individuals):
    """
    Removes duplicate individuals from the list.
    
    Args:
    individuals (list of lists): The list containing individuals (DNA sequences).

    Returns:
    list of lists: A list of unique individuals.
    """
    # Use a set to track unique individuals
    unique_individuals = set(tuple(individual) for individual in individuals)
    
    # Convert back to a list of lists
    return [list(indiv) for indiv in unique_individuals]

if(__name__ == '__main__'):
    # Suppress all warnings in logging
    logging.getLogger().setLevel(logging.ERROR)

    # Temporarily suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        solve(1000)