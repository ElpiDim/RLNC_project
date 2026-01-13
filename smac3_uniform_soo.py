#!/usr/bin/env python3
# encoding: utf-8

from smac import HyperparameterOptimizationFacade as RFfacade
from smac import BlackBoxFacade as GPfacade
from smac import RandomFacade as RSfacade
from smac import Scenario
from smac.initial_design.random_design import RandomInitialDesign

from ConfigSpace import ConfigurationSpace, Integer, EqualsCondition, Configuration

import numpy as np
import math
import os
import random
import argparse
import time
import faulthandler
import logging

import pyerasure.sw as pysw
import pyerasure.finite_field
import pyerasure.sw.generator as pygenerator

from state_vector_utils import StateVectorUtils

# Suppress all SMAC logs
logger = logging.getLogger("smac")
logger.setLevel(logging.ERROR)

# Globals for pretty printing
EVAL_COUNTER = 0
NUM_TRIALS_GLOBAL = 0

seen_points = set()
reward_terms = {}

MAX_REWARD = 2.0
EPS_COST = 1e-8


def safe_float_tag(x: float) -> str:
    # Safe string for filenames: 0.99 -> "0p99"
    return f"{x}".replace(".", "p")


def run_simulation(r, d, extraparams):
    """
    Runs one simulation and returns:
      reward_g, delivery_ratio, avg_inorder_delay
    """
    code_rate = r
    coding_depth = d
    field = extraparams[0]
    num_packets = extraparams[1]
    packet_size = extraparams[2]
    ploss = extraparams[3]
    D = extraparams[4]
    DRT = extraparams[5]
    seed = extraparams[-1]

    source_per_cycle = code_rate[0]
    repair_per_cycle = code_rate[1] - code_rate[0]
    source_plus_repair = code_rate[1]
    rng_loss = random.Random(seed + 1000)

    encoder = pysw.Encoder(field, num_packets, packet_size, coding_depth, source_per_cycle, repair_per_cycle)
    decoding_depth = coding_depth
    decoder = pysw.Decoder(field, num_packets, packet_size, coding_depth, source_per_cycle, decoding_depth)

    generator = pygenerator.RandomUniform(field, encoder.packets)
    generator.set_seed(seed + 2000)

    data_in = bytearray(random.Random(seed).getrandbits(8) for _ in range(num_packets * packet_size))
    encoder.set_packets(data_in)

    source_packet_counter = 0
    transmitted_packets = 0
    packets_sent = 0
    elapsed_timeslots = -1

    for index in range(num_packets):
        source_packet_counter += 1
        packet = encoder.packet_data(index)
        transmitted_packets += 1
        encoder.update_coding_window(index)
        elapsed_timeslots += 1

        decoder.update_timestamps(index, elapsed_timeslots)

        if rng_loss.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_systematic_packet(packet, index)
        packets_sent += 1

        if source_packet_counter % encoder.source_per_cycle == 0:
            for _ in range(encoder.repair_per_cycle):
                non_zero_coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
                zeros_before = bytearray(encoder.coding_window_start)
                zeros_after = bytearray(num_packets - encoder.coding_window_end)
                coefficients = zeros_before + non_zero_coefficients + zeros_after
                packet = encoder.encode_packet(coefficients)
                elapsed_timeslots += 1

                if rng_loss.uniform(0, 1) >= ploss:
                    decoder.current_timeslot = elapsed_timeslots
                    decoder.decode_packet(
                        packet,
                        bytearray(coefficients),
                        (encoder.coding_window_start, encoder.coding_window_end - 1),
                    )

    repair_at_the_end = math.ceil(
        (num_packets * source_plus_repair / source_per_cycle)
        - num_packets
        - (math.floor(num_packets / source_per_cycle) * repair_per_cycle)
    )

    for _ in range(repair_at_the_end):
        non_zero_coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
        zeros_before = bytearray(encoder.coding_window_start)
        zeros_after = bytearray(num_packets - encoder.coding_window_end)
        coefficients = zeros_before + non_zero_coefficients + zeros_after
        packet = encoder.encode_packet(coefficients)
        elapsed_timeslots += 1

        if rng_loss.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_packet(
                packet,
                bytearray(coefficients),
                (encoder.coding_window_start, encoder.coding_window_end - 1),
            )

    decoder.current_timeslot = elapsed_timeslots
    decoder.sum_delay_of_last()

    assert packets_sent == transmitted_packets

    delivery_ratio = decoder.delivered_packets / transmitted_packets if transmitted_packets else 0.0

    if decoder.delivered_packets > 0:
        avg_inorder_delay = decoder.sum_of_delay / decoder.delivered_packets
    else:
        avg_inorder_delay = float("inf")

    if not np.isfinite(avg_inorder_delay):
        avg_inorder_delay = float("inf")

    # reward g(D,DRT)
    if avg_inorder_delay <= D and delivery_ratio >= DRT:
        reward_g = (D - avg_inorder_delay) / D + delivery_ratio
    else:
        reward_g = 0.0

    return [reward_g, delivery_ratio, avg_inorder_delay]


def objective_function(config: Configuration, seed: int, extra_input, num_iterations: int):
    """
    Returns NON-NEGATIVE cost for SMAC (minimization).
    cost = MAX_REWARD - mean(g)
    """
    global EVAL_COUNTER, NUM_TRIALS_GLOBAL
    EVAL_COUNTER += 1

    index = config["rate_index"]
    rate = StateVectorUtils.valid_rates[index]
    possible_d = StateVectorUtils.calculate_possible_d_values(rate[0])

    d_index = config[f"d_{index}"]
    d = possible_d[d_index]

    # ---- PRINT EVALUATION COUNTER ----
    if NUM_TRIALS_GLOBAL > 0:
        print(f"\n=== Evaluation {EVAL_COUNTER}/{NUM_TRIALS_GLOBAL} ===")
    else:
        print(f"\n=== Evaluation {EVAL_COUNTER} ===")

    print("Config:", dict(config))
    print(f"Input: R={rate}, d={d}")

    seen_points.add((rate, d))

    outputs = [
        run_simulation(rate, d, extra_input[:] + [seed + i])
        for i in range(num_iterations)
    ]

    rewards = [el[0] for el in outputs]
    delivery_ratios = [el[1] for el in outputs]
    avg_delays = [el[2] for el in outputs]

    mean_reward = float(np.mean(rewards))
    mean_dr = float(np.mean(delivery_ratios))
    mean_delay = float(np.mean(avg_delays))

    reward_terms[(rate, d)] = (mean_dr, mean_delay)

    cost = float(MAX_REWARD - mean_reward)

    if not np.isfinite(cost):
        cost = MAX_REWARD
    if cost <= 0.0:
        cost = EPS_COST

    return cost


def main():
    global EVAL_COUNTER, NUM_TRIALS_GLOBAL
    EVAL_COUNTER = 0

    faulthandler.enable()

    parser = argparse.ArgumentParser()

    # Simulation args
    parser.add_argument("--num_packets", type=int, default=1000)
    parser.add_argument("--packet_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--ploss", type=float, default=0.1)

    # SMAC args
    parser.add_argument("--random_percentage", type=float, default=0.3)
    parser.add_argument("--n_calls", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--estimator", type=str, default="RF")
    parser.add_argument("--state_space", type=str, default="short")

    # Targets
    parser.add_argument("--D", type=int, required=True, help="Delay threshold in slots")
    parser.add_argument("--DRT", type=float, required=True, help="Delivery ratio threshold (e.g. 0.99)")

    args = parser.parse_args()

    num_packets = args.num_packets
    packet_size = args.packet_size
    seed = args.seed
    ploss = args.ploss
    D = args.D
    DRT = args.DRT

    estimator = args.estimator
    if estimator not in ("Random", "RF", "GP"):
        raise ValueError("Estimator type not supported. Use Random, RF, or GP.")

    random_percentage = args.random_percentage
    num_trials = args.n_calls
    num_iterations = args.iterations
    NUM_TRIALS_GLOBAL = num_trials

    # Init search space
    if args.state_space == "short":
        StateVectorUtils.initialize_state_space_short()
    else:
        StateVectorUtils.initialize_state_space_full()

    print("SMAC Training using PyErasure")
    print("---------------------------------------")
    print(f"Number of Trials:\t\t{num_trials}")
    print(f"Number of Packets/Trial:\t{num_packets}")
    print(f"Packet Size (bytes):\t\t{packet_size}")
    print(f"Ploss:\t\t\t\t{ploss*100}%")
    print(f"D (delay thr):\t\t\t{D} slots")
    print(f"DRT (DR thr):\t\t\t{DRT}")
    print(f"Seed:\t\t\t\t{seed}")
    print(f"State Space Size:\t\t{StateVectorUtils.state_space_size()}")

    field = pyerasure.finite_field.Binary8()

    # valid rates filtered by channel constraint
    StateVectorUtils.valid_rates = [r for r in StateVectorUtils.get_code_rates() if r[0] / r[1] <= 1 - ploss]
    if len(StateVectorUtils.valid_rates) == 0:
        raise RuntimeError("No valid rates after filtering with constraint r0/r1 <= 1-ploss. Check ploss/state_space.")

    params = [field, num_packets, packet_size, ploss, D, DRT]

    def obj_func(config: Configuration, seed: int):
        return objective_function(config, seed, extra_input=params, num_iterations=num_iterations)

    # Config space (new API)
    cs = ConfigurationSpace(seed=seed + 1)

    rate_index = Integer("rate_index", bounds=(0, len(StateVectorUtils.valid_rates) - 1))
    cs.add(rate_index)

    for idx, rate in enumerate(StateVectorUtils.valid_rates):
        possible_d = StateVectorUtils.calculate_possible_d_values(rate[0])
        if len(possible_d) == 0:
            raise RuntimeError(f"possible_d is empty for rate={rate}. Check calculate_possible_d_values().")

        d_hp = Integer(f"d_{idx}", bounds=(0, len(possible_d) - 1))
        cs.add(d_hp)

        cond = EqualsCondition(d_hp, rate_index, idx)
        cs.add(cond)

    # Root output folder
    results_root = "examples/bo/results/soo/smac"
    os.makedirs(results_root, exist_ok=True)

    # Unique run dir (avoid resume/caching weirdness)
    ts = time.strftime("%Y%m%d-%H%M%S")
    drt_tag = safe_float_tag(DRT)
    ploss_tag = safe_float_tag(ploss)

    run_dir = os.path.join(
        results_root,
        f"n{num_trials}_D{D}_DRT{drt_tag}_ploss{ploss_tag}_pkts{num_packets}_"
        f"{args.state_space}_{estimator}_seed{seed}_it{num_iterations}_{ts}"
    )
    os.makedirs(run_dir, exist_ok=True)

    scenario = Scenario(
        cs,
        n_trials=num_trials,
        seed=seed + 2,
        deterministic=True,  # keep as you requested
        n_workers=1,
        output_directory=run_dir,
    )

    initial_design = RandomInitialDesign(
        scenario=scenario,
        n_configs=max(1, int(random_percentage * num_trials)),
        max_ratio=1.0,
    )

    start = time.time()
    if estimator == "Random":
        smac = RSfacade(scenario, obj_func)
    elif estimator == "RF":
        smac = RFfacade(scenario=scenario, target_function=obj_func, initial_design=initial_design)
    else:
        smac = GPfacade(scenario=scenario, target_function=obj_func, initial_design=initial_design)

    incumbent = smac.optimize()
    end = time.time()

    # Best config (incumbent)
    cfg = incumbent
    best_rate_index = cfg["rate_index"]
    best_rate = StateVectorUtils.valid_rates[best_rate_index]
    best_depths = StateVectorUtils.calculate_possible_d_values(best_rate[0])
    best_d_index = cfg[f"d_{best_rate_index}"]
    best_d = best_depths[best_d_index]

    # metrics from stored evaluations
    best_dr, best_delay = reward_terms.get((best_rate, best_d), (float("nan"), float("nan")))

    # reward g from mean metrics
    if np.isfinite(best_delay) and np.isfinite(best_dr) and best_delay <= D and best_dr >= DRT:
        best_reward = (D - best_delay) / D + best_dr
    else:
        best_reward = 0.0

    runtime_sec = float(end - start)

    print(f"\nBest state: {best_rate}, d={best_d}, Reward(g)={best_reward:.6f}")
    print(
        f"  Mean DR={best_dr:.6f}, Mean Delay={best_delay:.6f}, "
        f"Time={runtime_sec:.2f}s, UniquePoints={len(seen_points)}"
    )

    # ---- SAVE RESULTS AS TABLE ROW ----
    results_file = os.path.join(
        results_root,
        f"smac_summary.tsv"
    )

    header = (
        "model\tEstimator\tn_calls\tploss\tStateSpace\tSeed\tD\tDRT\tBest R\tBest D\t"
        "Best Score g\tAvg delivery ratio\tavg inorder delay\truntime (sec)\n"
    )

    row = (
        f"SMAC\t{estimator}\t{num_trials}\t{ploss}\t{args.state_space}\t{seed}\t{D}\t{DRT}\t"
        f"{best_rate[0]}/{best_rate[1]}\t{best_d}\t{best_reward:.6f}\t"
        f"{best_dr:.6f}\t{best_delay:.6f}\t{runtime_sec:.2f}\n"
    )

    write_header = not os.path.exists(results_file) or os.path.getsize(results_file) == 0
    with open(results_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(row)

    print(f"\nSaved summary row to: {results_file}")
    print(f"SMAC run directory: {run_dir}")


if __name__ == "__main__":
    main()
