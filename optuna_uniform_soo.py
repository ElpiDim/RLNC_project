#!/usr/bin/env python
# encoding: utf-8

import math
import os
import random
import argparse
import time
import faulthandler

import numpy as np
import optuna
from optuna import TrialPruned
from functools import partial
from pyerasure.sw.swencoder import SWEncoder
from pyerasure.sw.swdecoder import SWDecoder

import pyerasure.sw as pysw
import pyerasure.finite_field
import pyerasure.sw.generator as pygenerator

from state_vector_utils import StateVectorUtils


# Optimization script for channels with i.i.d losses using Optuna's Random Sampler and Tree Parzen Estimator (TPE) Sampler.

MAX_CW_SIZE = 200
seen_points = set()
reward_terms = {}

def run_simulation(r, d, extraparams):
    # A complete PyErasure simulation - the objective function
    code_rate = r
    coding_depth = d
    field = extraparams[0]
    num_packets = extraparams[1]
    packet_size = extraparams[2]
    ploss = extraparams[3]
    code_rate_weight = extraparams[4]
    window_weight = extraparams[5]
    reward_func = extraparams[6]
    D = extraparams[7]
    DRT = extraparams[8]
    seed = extraparams[-1]

    source_per_cycle = code_rate[0]
    repair_per_cycle = code_rate[1] - code_rate[0]
    source_plus_repair = code_rate[1]
    rng_loss = random.Random(seed + 1000) # generator for channel losses
    # Create an encoder, a decoder and a generator.
    encoder = pysw.SWEncoder(field, num_packets, packet_size, coding_depth, source_per_cycle, repair_per_cycle)
    decoding_depth = coding_depth # assume CW = DW
    decoder = pysw.SWDecoder(field, num_packets, packet_size, coding_depth, source_per_cycle, decoding_depth)
    generator = pygenerator.RandomUniform(field, encoder.packets)
    generator.set_seed(seed + 2000)

    # Initialize data buffer (we'll feed it per packet using set_packet)
    data_in = bytearray(random.Random(seed).getrandbits(8) for _ in range(num_packets * packet_size))


    source_packet_counter = 0 # counter to decide when to send repair packets - if complete division, time for coded pkts
    transmitted_packets = 0 # counter for the total number of source packets sent from the encoder
    packets_sent = 0 # Counter for the total source packets sent during the episode - also index of current src packet
    repair_packets = 0 # total number of repair packets transmitted during the episode
    elapsed_timeslots = -1 # indicating how many timeslots have elapsed since the beginning of transmission

    # start = time.time()
    for index in range(num_packets):
        source_packet_counter += 1
        offset = index * encoder.packet_size_bytes
        encoder.set_packet(index, data_in[offset : offset + encoder.packet_size_bytes])
        packet = encoder.packet_data(index)
        transmitted_packets += 1

        elapsed_timeslots += 1

        decoder.update_timestamps(index, elapsed_timeslots)

        if rng_loss.uniform(0,1) >= ploss: 
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_systematic_packet(packet, index)
        packets_sent += 1

        #send |repair_per_cycle| repair packets every |source_per_cycle| source ones
        if source_packet_counter % encoder.source_per_cycle == 0:
            for r in range(encoder.repair_per_cycle):
                #assert packets_sent != 0 #should never enter here if 0 coefficients are to be generated
                
                coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
                packet = encoder.encode_packet(coefficients)
                repair_packets += 1
                elapsed_timeslots += 1
                
                if rng_loss.uniform(0,1) >= ploss: 
                    decoder.current_timeslot = elapsed_timeslots
                    decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end-1))
    repair_at_the_end = math.ceil((num_packets*source_plus_repair / source_per_cycle) - num_packets - (math.floor(num_packets/source_per_cycle)*repair_per_cycle))
    for _ in range(repair_at_the_end):
        #print(f"Repair packet {r+1} out of {repair_at_the_end} at slot {elapsed_timeslots+1}.")

        coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
        packet = encoder.encode_packet(coefficients)
        repair_packets += 1
        elapsed_timeslots += 1
        if rng_loss.uniform(0,1) >= ploss: 
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end-1))

    decoder.current_timeslot = elapsed_timeslots
    decoder.sum_delay_of_last()    
    # end = time.time()

    assert packets_sent == transmitted_packets 

    delivery_ratio = decoder.delivered_packets / transmitted_packets if transmitted_packets else 0.0
    average_inorder_delay = (decoder.sum_of_delay / decoder.delivered_packets) if decoder.delivered_packets else float("inf")

    #neo g (D, DRT)
    if average_inorder_delay <= D and delivery_ratio >= DRT:
        reward_score = (D - average_inorder_delay)/ D + delivery_ratio
    else:
        reward_score = 0.0 
    #minr = StateVectorUtils.min_code_rate()
    #r_min = minr[0]/minr[1]
    #maxr = StateVectorUtils.max_code_rate()
    #r_max = maxr[0]/maxr[1]
    #r = code_rate[0]/code_rate[1]
    #r_term = (r_max - r)/(r_max - r_min)
    #w_min = StateVectorUtils.min_cw_size()
   # w_term = (coding_window_size - w_min)/(MAX_CW_SIZE - w_min)
   # match reward_func:
    #    case "linear":
     #       reward_score = delivery_ratio - code_rate_weight*r_term - window_weight*w_term
     #   case "nonlinear":
     #       reward_score = delivery_ratio - code_rate_weight*math.pow(r_term, 2) - window_weight*math.pow(w_term, 2)
    
    #output = [reward_score, delivery_ratio, r_term + w_term]
    output = [reward_score, delivery_ratio, average_inorder_delay] # dn mas endiaferei to complexity alla to delay 
    return output

def objective_function(trial: optuna.Trial, extra_input, num_iterations):
    """
    Called by the optimizer. Sets up and runs simulation for each (r, d) for |num_iterations| times.
    Additional parameters are passed in from the user.
    """

    rate_list = StateVectorUtils.valid_rates
    index = trial.suggest_int("rate", 0, len(rate_list)-1)
    rate = rate_list[index]
    max_d = MAX_CW_SIZE // rate[0]

    possible_d = StateVectorUtils.calculate_possible_d_values(rate[0])
    d_index = trial.suggest_int("d", 0, len(possible_d)-1)
    d = possible_d[d_index]

    print(f"Input: R={rate}, d={d}")
    if d > max_d:
        raise ValueError(f"Invalid value for d={d} > {max_d}")
    if (rate, d) in seen_points:
        raise TrialPruned()
    seen_points.add((rate, d))

    # --- Simulate delivery ratio ---
    outputs = [run_simulation(rate, d, extra_input[:] + [i+1]) for i in range(num_iterations)]

    scores = [el[0] for el in outputs]
    delivery_ratios = [el[1] for el in outputs]
    #complexity = outputs[0][2]
    #if (rate, d) not in reward_terms:
    #    reward_terms[(rate, d)] = (np.mean(delivery_ratios), complexity)
    avg_delay = np.mean([el[2] for el in outputs])
    if (rate, d) not in reward_terms:
        reward_terms[(rate, d)] = (np.mean(delivery_ratios), avg_delay)

    avg_score = np.mean(scores)
    return avg_score 

def custom_gamma(x: int, gamma: float) -> int:
    return min(int(np.ceil(gamma * x)), 25)

def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()

    # Simulation related arguments
    parser.add_argument("--num_packets", default = 1000, required = False, help="Total number of source packets transmitted per trial")
    parser.add_argument("--code_rate_weight", default = 0.2, required = False, help="Weight indicating the impact of R on the reward function")
    parser.add_argument("--window_weight", default = 0.2, required = False, help="Weight indicating the impact of CW size on the reward function")    
     
    # Simulation-specific arguments
    parser.add_argument("--packet_size", default = 100, required = False, help="Packet size in bytes")
    parser.add_argument("--seed", default = 5, required = False, help="Seed to control randomness and reproducibility")
    parser.add_argument("--ploss", default = 0.1, required = False, help="Channel's average packet loss probability (in range [0,1])")

    parser.add_argument("--random_percentage", default = 0.3, required = False, help = "Percentage of random evaluations before fitting the model. Not applied in RandomSearch.")
    parser.add_argument("--n_calls", default = 10, required = False, help="Total number of function evaluations per training")
    parser.add_argument("--iterations", default = 5, required = False, help="Number of iterations per set of coding parameters")
    parser.add_argument("--estimator", default = "TPE", required = False, help = "Option for the optimizer (sampler), either TPE or Random.")
    parser.add_argument("--func", default = "linear", required = False, help = "linear and nonlinear are the possible options.")
    parser.add_argument("--gamma", default = 0.1, required = False, help="Ratio of good configurations for TPE, determines exploration-exploitation trade-off.")
    parser.add_argument("--state_space", default = "short", required = False, help = "short or full search space are the possible options.")

    parser.add_argument("--D", type=int, required=True, help="Delay threshold in slots")
    parser.add_argument("--DRT", type=float, required=True, help="Delivery ratio threshold (e.g. 0.99)")

    args = parser.parse_args()
    D = int(args.D)
    DRT = float(args.DRT)


    # Initialize given arguments
    num_packets = int(args.num_packets)
    code_rate_weight = float(args.code_rate_weight)
    window_weight = float(args.window_weight)

    packet_size = int(args.packet_size)
    seed = int(args.seed)
    ploss = float(args.ploss)

    estimator = args.estimator
    if not (estimator == "Random" or estimator == "TPE"):
        raise ValueError("Estimator type not supported.")
    random_percentage = float(args.random_percentage)
    num_trials = int(args.n_calls)
    num_iterations = int(args.iterations)
    if estimator == "TPE":
        gamma = float(args.gamma)
    
    # Initialize available coding depth values per code rate
    if args.state_space == "short":
        StateVectorUtils.initialize_state_space_short()
    else:
        StateVectorUtils.initialize_state_space_full()
    
    print(f"Optuna Training using PyErasure\n---------------------------------------")
    print(f"Number of Trials: \t\t{num_trials}\n"
          f"Number of Packets/Trial: \t{num_packets}\n"
          f"Packet Size (bytes): \t\t{packet_size}\n"
          f"Ploss: \t\t\t\t{ploss*100}%\n"
          f"R Weight: \t\t\t{code_rate_weight}\n"
          f"CW Weight: \t\t\t{window_weight}\n"
          f"Seed: \t\t\t\t{seed}\n")

    field = pyerasure.finite_field.Binary8()

    random_evaluations = int(random_percentage*num_trials)
    if estimator == "Random":
        random_txt = "_" + args.func
    else:
        random_txt = "_random" + str(random_percentage) + "_" + args.func + "_gamma" + str(gamma)

    path = "examples/bo/results/soo"
    if not os.path.exists(path):
        os.makedirs(path)

    StateVectorUtils.valid_rates = [r for r in StateVectorUtils.get_code_rates() if r[0]/r[1] <= 1-ploss]   
    params = [field, num_packets, packet_size, ploss, code_rate_weight, window_weight, args.func, D, DRT] # list with arguments needed from a PyErasure simulation
    obj_func = partial(objective_function, extra_input=params, num_iterations=num_iterations)

    start = time.time()
    if estimator == "Random":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed))
    else:
        gamma_func = partial(custom_gamma, gamma=gamma) 
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=random_evaluations, gamma=gamma_func))
    
    # print(StateVectorUtils.valid_rates)
    study.optimize(obj_func, n_trials=num_trials)
    end = time.time()

    results_filename = path + "/ploss" + str(ploss) + "_pkts" + str(num_packets) + random_txt + "_" + args.state_space + "_" + estimator + "_rw" + str(code_rate_weight) + "_ww" + str(window_weight) + "_results.txt"
    with open(results_filename, 'a', encoding='UTF8') as f:
        rate_index = study.best_params['rate']
        best_r = StateVectorUtils.valid_rates[rate_index]
        d_index = study.best_params['d']
        possible_d = StateVectorUtils.calculate_possible_d_values(best_r[0])
        d = possible_d[d_index]

        best_avg_terms = reward_terms[(best_r, d)]
        f.write(
        f"{estimator} Sampler - Seed: {seed} - Iterations: {num_iterations} - Number of trials: {num_trials} "
        f"- D={D} - DRT={DRT} - ploss={ploss} - state_space={args.state_space} "
        f"- Best (R,d)=({best_r[0]}/{best_r[1]}, {d}) "
        f"- Score={study.best_value:.6f} "
        f"- DR={best_avg_terms[0]:.6f} - AvgDelay={best_avg_terms[1]:.6f} "
        f"- Time={end-start:.2f}s - UniquePoints={len(seen_points)}\n"
    )


if __name__ == "__main__":
    main()
