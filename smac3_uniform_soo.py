#!/usr/bin/env python3
# encoding: utf-8

import math
import os
import random
import argparse
import time
import faulthandler
import logging
import numpy as np

from smac import HyperparameterOptimizationFacade as RFfacade
from smac import BlackBoxFacade as GPfacade
from smac import RandomFacade as RSfacade
from smac import Scenario
from smac.initial_design.random_design import RandomInitialDesign
from ConfigSpace import ConfigurationSpace, Integer, EqualsCondition, Configuration

from pyerasure.sw.swencoder import SWEncoder
from pyerasure.sw.swdecoder import SWDecoder
import pyerasure.sw as pysw
import pyerasure.finite_field
import pyerasure.sw.generator as pygenerator
from state_vector_utils import StateVectorUtils

# Καταστολή logs του SMAC για καθαρότερο terminal
logger = logging.getLogger("smac")
logger.setLevel(logging.ERROR)

# --- Σταθερές & Παράμετροι Delay (Από Optuna) ---
MAX_CW_SIZE = 200
SLOT_US = 125.0  # slot duration in microseconds
A, B, C = 0.0501, -0.0926, 17.652
K_REF, TOTAL_REF = 9, 10
N_REF = TOTAL_REF - K_REF

MAX_REWARD = 2.0
EPS_COST = 1e-8
EVAL_COUNTER = 0
NUM_TRIALS_GLOBAL = 0
seen_points = set()
reward_terms = {}

def processing_delay_slots(code_rate, coding_depth):
    """Υπολογισμός του processing delay σε slots."""
    k = code_rate[0]
    total = code_rate[1]
    n = total - k
    w = coding_depth * k
    d_us = A * (w ** 2) + B * w + C
    scale = (n * K_REF) / (N_REF * k)
    d_us_scaled = d_us * scale
    return d_us_scaled / SLOT_US

def run_simulation(r, d, extraparams):
    """Προσομοίωση PyErasure με τη συμμετρική reward function του Optuna."""
    code_rate = r
    coding_depth = d
    field, num_packets, packet_size, ploss = extraparams[0:4]
    D, DRT = extraparams[7], extraparams[8]
    seed = extraparams[-1]

    source_per_cycle = code_rate[0]
    repair_per_cycle = code_rate[1] - code_rate[0]
    source_plus_repair = code_rate[1]
    
    rng_loss = random.Random(seed + 1000)
    encoder = SWEncoder(field, num_packets, packet_size, coding_depth, source_per_cycle, repair_per_cycle)
    decoder = SWDecoder(field, num_packets, packet_size, coding_depth, source_per_cycle, coding_depth)
    generator = pygenerator.RandomUniform(field, encoder.packets)
    generator.set_seed(seed + 2000)

    data_in = bytearray(random.Random(seed).getrandbits(8) for _ in range(num_packets * packet_size))
    source_packet_counter = 0
    transmitted_packets = 0
    elapsed_timeslots = -1

    for index in range(num_packets):
        source_packet_counter += 1
        offset = index * encoder.packet_size_bytes
        encoder.set_packet(index, data_in[offset : offset + encoder.packet_size_bytes])
        packet = encoder.packet_data(index)
        transmitted_packets += 1
        elapsed_timeslots += 1
        decoder.update_timestamps(index, elapsed_timeslots)

        if rng_loss.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_systematic_packet(packet, index)

        if source_packet_counter % encoder.source_per_cycle == 0:
            for _ in range(encoder.repair_per_cycle):
                coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
                packet = encoder.encode_packet(coefficients)
                elapsed_timeslots += 1
                if rng_loss.uniform(0, 1) >= ploss:
                    decoder.current_timeslot = elapsed_timeslots
                    decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end - 1))

    # Repair packets στο τέλος
    repair_at_the_end = math.ceil((num_packets * source_plus_repair / source_per_cycle) - num_packets - (math.floor(num_packets / source_per_cycle) * repair_per_cycle))
    for _ in range(repair_at_the_end):
        coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
        packet = encoder.encode_packet(coefficients)
        elapsed_timeslots += 1
        if rng_loss.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end - 1))

    decoder.current_timeslot = elapsed_timeslots
    decoder.sum_delay_of_last()

    delivery_ratio = decoder.delivered_packets / transmitted_packets if transmitted_packets else 0.0
    avg_inorder_delay = (decoder.sum_of_delay / decoder.delivered_packets) if decoder.delivered_packets else float("inf")
    
    proc_delay = processing_delay_slots(code_rate, coding_depth)
    total_delay = avg_inorder_delay + proc_delay

    # old reward function
    # if total_delay <= D and delivery_ratio >= DRT:
    #     reward_score = (D - total_delay) / D + (delivery_ratio - DRT) / (1.0 - DRT)
    # else:
    #     reward_score = 0.0

    # new reward function g2
    R = code_rate[0] / code_rate[1]

    if total_delay <= D and delivery_ratio >= DRT:
        reward_score = (D - total_delay) / D + R * ((delivery_ratio - DRT) / (1.0 - DRT))
    else:
        reward_score = 0.0

    return [reward_score, delivery_ratio, total_delay]

def objective_function(config, seed, extra_input, num_iterations):
    global EVAL_COUNTER, NUM_TRIALS_GLOBAL
    EVAL_COUNTER += 1
    
    rate_idx = config["rate_index"]
    rate = StateVectorUtils.valid_rates[rate_idx]
    possible_d = StateVectorUtils.calculate_possible_d_values(rate[0])
    d = possible_d[config[f"d_{rate_idx}"]]

    if NUM_TRIALS_GLOBAL > 0:
        print(f"Evaluation {EVAL_COUNTER}/{NUM_TRIALS_GLOBAL} -> R={rate}, d={d}")

    outputs = [run_simulation(rate, d, extra_input[:] + [seed + i]) for i in range(num_iterations)]
    
    mean_reward = np.mean([el[0] for el in outputs])
    reward_terms[(rate, d)] = (np.mean([el[1] for el in outputs]), np.mean([el[2] for el in outputs]))
    
    return float(MAX_REWARD - mean_reward) # SMAC ελαχιστοποιεί το cost

def main():
    global NUM_TRIALS_GLOBAL
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_packets", type=int, default=1000)
    parser.add_argument("--packet_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--ploss", type=float, default=0.1)
    parser.add_argument("--n_calls", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--estimator", type=str, default="RF") # RF, GP, Random
    parser.add_argument("--state_space", type=str, default="short")
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--DRT", type=float, required=True)
    # Args για συμβατότητα με Optuna command line
    parser.add_argument("--code_rate_weight", type=float, default=0.2)
    parser.add_argument("--window_weight", type=float, default=0.2)
    parser.add_argument("--func", type=str, default="linear")
    parser.add_argument("--random_percentage", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)

    args = parser.parse_args()
    NUM_TRIALS_GLOBAL = args.n_calls

    if args.state_space == "short": StateVectorUtils.initialize_state_space_short()
    else: StateVectorUtils.initialize_state_space_full()

    StateVectorUtils.valid_rates = [r for r in StateVectorUtils.get_code_rates() if r[0]/r[1] <= 1 - args.ploss]
    field = pyerasure.finite_field.Binary8()
    params = [field, args.num_packets, args.packet_size, args.ploss, args.code_rate_weight, args.window_weight, args.func, args.D, args.DRT]

    cs = ConfigurationSpace(seed=args.seed)
    rate_hp = Integer("rate_index", (0, len(StateVectorUtils.valid_rates) - 1))
    cs.add(rate_hp)
    for i, r in enumerate(StateVectorUtils.valid_rates):
        d_vals = StateVectorUtils.calculate_possible_d_values(r[0])
        d_hp = Integer(f"d_{i}", (0, len(d_vals) - 1))
        cs.add(d_hp)
        cs.add(EqualsCondition(d_hp, rate_hp, i))

    scenario = Scenario(cs, n_trials=args.n_calls, seed=args.seed, output_directory="smac_logs")
    init_design = RandomInitialDesign(scenario, n_configs=max(1, int(args.random_percentage * args.n_calls)))

    if args.estimator == "Random":
        smac = RSfacade(
            scenario,
            lambda config, seed: objective_function(config, seed, params, args.iterations)
        )
    elif args.estimator == "RF":
        smac = RFfacade(
            scenario,
            lambda config, seed: objective_function(config, seed, params, args.iterations),
            initial_design=init_design
        )
    else:
        smac = GPfacade(
            scenario,
            lambda config, seed: objective_function(config, seed, params, args.iterations),
            initial_design=init_design
        )
    start = time.time()
    incumbent = smac.optimize()
    runtime = time.time() - start

    # --- Extraction & Saving ---
    best_r = StateVectorUtils.valid_rates[incumbent["rate_index"]]
    best_d = StateVectorUtils.calculate_possible_d_values(best_r[0])[incumbent[f"d_{incumbent['rate_index']}"]]
    best_metrics = reward_terms[(best_r, best_d)]
    
    res_path = "examples/bo/results/soo"
    os.makedirs(res_path, exist_ok=True)
    res_file = os.path.join(res_path, "smac_summary.tsv")

    header = "Estimator\tN_calls\titerations\tploss\tnum_packets\tseed\tD\tDRT\tBest R\tBest D\tBest Score g\tAvg delivery ratio\ttotal delay\truntime (sec)\n"
    row = f"{args.estimator}\t{args.n_calls}\t{args.iterations}\t{args.ploss}\t{args.num_packets}\t{args.seed}\t{args.D}\t{args.DRT}\t{best_r[0]}/{best_r[1]}\t{best_d}\t{MAX_REWARD - smac.validate(incumbent):.6f}\t{best_metrics[0]:.6f}\t{best_metrics[1]:.6f}\t{runtime:.2f}\n"

    with open(res_file, "a") as f:
        if not os.path.exists(res_file) or os.path.getsize(res_file) == 0:
            f.write(header)
        f.write(row)
    print(f"Finished! Results saved to {res_file}")

if __name__ == "__main__":
    main()
