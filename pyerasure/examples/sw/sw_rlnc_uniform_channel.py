#!/usr/bin/env python
# encoding: utf-8

# License for Commercial Usage
# Distributed under the "PYERASURE EVALUATION LICENSE 1.3"
# Licensees holding a valid commercial license may use this project in
# accordance with the standard license agreement terms provided with the
# Software (see accompanying file LICENSE.rst or
# https://www.steinwurf.com/license), unless otherwise different terms and
# conditions are agreed in writing between Licensee and Steinwurf ApS in which
# case the license will be regulated by that separate written agreement.
#
# License for Non-Commercial Usage
# Distributed under the "PYERASURE RESEARCH LICENSE 1.2"
# Licensees holding a valid research license may use this project in accordance
# with the license agreement terms provided with the Software
# See accompanying file LICENSE.rst or https://www.steinwurf.com/license

import math
import random
import argparse
import time
import csv 

import pyerasure.sw as pysw
import pyerasure.finite_field
import pyerasure.sw.generator as pygenerator

# Custom script to collect performance data for i.i.d channels.
# Code Rate and Coding Depth are required, the rest of the arguments are  optional.

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_packets", default = 20000, required = False, help = "Total number of source packets transmitted per episode")
    parser.add_argument("--packet_size", default = 1000, required = False, help = "Packet size in bytes")
    parser.add_argument("--coding_depth", help = "Coding depth value")
    parser.add_argument("--code_rate", help = "Code Rate in form k/n, where k is the number of source packets transmitted and n the sum of source and repair packets per cycle")
    parser.add_argument("--decoding_depth", default = 1, required = False, help = "Parameters to tune decoding window size")
    parser.add_argument("--seed", default = 5, required = False, help = "Seed to control randomness")
    parser.add_argument("--ploss", default = 0.1, required = False, help = "Channel's average packet loss probability (in range [0,1])")

    args = parser.parse_args()

    # Set the number of packets to encode/decode.
    packets = int(args.num_packets)

    # Set the size of each packet in bytes
    packet_bytes = int(args.packet_size)

    # Choose a coding depth value for the simulation
    coding_depth = int(args.coding_depth)

    # Define a code rate R using variables source_per_cycle and repair_per_cycle
    try:
        source_per_cycle, source_plus_repair = map(int, args.code_rate.split('/'))
        if source_plus_repair - source_per_cycle < 0:
            raise ValueError("Code Rate should be in range [0,1]. Please insert appropriate values.")
        repair_per_cycle = source_plus_repair - source_per_cycle
    except ValueError:
        raise ValueError("Code Rate shall be given in the given format.")

    # Choose a decoding window value for the decoding matrix size - here we assume CW=DW
    decoding_depth = coding_depth

    # Seed for randomness and reproducability
    seed = int(args.seed)

    # Average Loss probability given as a float in range [0,1]
    ploss = float(args.ploss)
    
    # Pick the finite field to use for the encoding and decoding.
    field = pyerasure.finite_field.Binary8()

    # Create an encoder, a decoder and a generator.
    encoder = pysw.Encoder(field, packets, packet_bytes, coding_depth, source_per_cycle, repair_per_cycle)

    # Generate Decoder based on input argument - 3 different implementations supported (Base, EED, EEDM)
    decoder = pysw.Decoder(field, packets, packet_bytes, coding_depth, source_per_cycle, decoding_depth)
        
    generator = pygenerator.RandomUniform(field, encoder.packets)
    generator.set_seed(seed)

    print(f"PyErasure Simulation\n---------------------------------------")
    print(f"Number of Transmitted Packets: \t{args.num_packets}\n"
          f"Packet Size (bytes): \t\t{args.packet_size}\n"
          f"Coding Depth: \t\t\t{args.coding_depth}\n"
          f"Coding Window Size: \t\t{encoder.coding_window_size}\n"
          f"Code Rate (R): \t\t\t{args.code_rate}\n"
          f"Decoding Depth: \t\t{decoding_depth}\n"
          f"Decoding Window Size: \t\t{decoder.decoding_window}\n"
          f"Loss Probability (%): \t\t{ploss*100}\n"
          f"Seed: \t\t\t\t{args.seed}")


    random.seed(seed)
    data_in = bytearray(random.getrandbits(8) for _ in range(packets*packet_bytes))

    # Assign the data buffer to the encoder
    encoder.set_packets(data_in)

    # counter to decide when to send repair packets
    source_packet_counter = 0

    # variables for statistics
    transmitted_packets = 0 # counter for the total number of source packets sent from the encoder
    repair_packets = 0 # total number of repair packets transmitted
    elapsed_timeslots = -1 # indicating how many timeslots have elapsed since the beginning of transmission
    
    start = time.time()
    for index in range(packets):
        source_packet_counter += 1
        packet = encoder.packet_data(index)
        transmitted_packets += 1
        encoder.update_coding_window(index)
        elapsed_timeslots += 1

        decoder.update_timestamps(index, elapsed_timeslots)

        if random.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_systematic_packet(packet, index)
 
        #send |repair_per_cycle| repair packets every |source_per_cycle| source ones
        if source_packet_counter % source_per_cycle == 0:
            for _ in range(repair_per_cycle):
                non_zero_coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
                zeros_before = bytearray(encoder.coding_window_start)
                zeros_after = bytearray(packets - encoder.coding_window_end)
                coefficients = zeros_before + non_zero_coefficients + zeros_after
                packet = encoder.encode_packet(coefficients)
                repair_packets += 1
                elapsed_timeslots += 1
                
                if random.uniform(0, 1) >= ploss:
                    decoder.current_timeslot = elapsed_timeslots
                    decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end-1))
    
    #send extra repair packets at the end if exists at least one source packet unprotected - with CW coefficients!
    repair_at_the_end = math.ceil((packets*source_plus_repair / source_per_cycle) - packets - (math.floor(packets/source_per_cycle)*repair_per_cycle))
    for _ in range(repair_at_the_end):
        non_zero_coefficients = generator.generate_partial(encoder.calculate_num_coefficients())
        zeros_before = bytearray(encoder.coding_window_start)
        zeros_after = bytearray(packets - encoder.coding_window_end)
        coefficients = zeros_before + non_zero_coefficients + zeros_after
        packet = encoder.encode_packet(coefficients)
        repair_packets += 1
        elapsed_timeslots += 1
        if random.uniform(0, 1) >= ploss:
            decoder.current_timeslot = elapsed_timeslots
            decoder.decode_packet(packet, bytearray(coefficients), (encoder.coding_window_start, encoder.coding_window_end-1))


    decoder.current_timeslot = elapsed_timeslots
    decoder.sum_delay_of_last()     
 
    end = time.time()

    print("#####-----Statistics-----#####")
    print(f"Total Transmitted packets: \t\t{transmitted_packets}")
    print(f"Total Delivered Packets: \t\t{decoder.delivered_packets}")
    print(f"Total Repair Packets: \t\t\t{repair_packets}")
    delivery_ratio = decoder.delivered_packets / transmitted_packets if transmitted_packets != 0 else 0
    formatted_delivery_ratio = "%.9f" % delivery_ratio # <class str>!!!
    print(f"Delivery Ratio: \t\t\t{formatted_delivery_ratio}")
    packet_drop_rate = 1 - (decoder.delivered_packets / transmitted_packets) if transmitted_packets != 0 else 1
    formatted_drop_rate = "%.9f" % packet_drop_rate # <class str>!!!
    print(f"Packet Drop Rate: \t\t\t{formatted_drop_rate}")
    avg_inorder_delay = decoder.sum_of_delay / decoder.delivered_packets if decoder.delivered_packets != 0 else 0
    print(f"Average In-order Delay: \t\t{avg_inorder_delay} slots")
    goodput = (source_per_cycle/source_plus_repair) * delivery_ratio
    print(f"Goodput: \t\t\t\t{goodput}")  
    avg_dm_size = decoder.dm_size / decoder.complete_decodings if decoder.complete_decodings != 0 else 0
    print(f"Average Decoding Matrix Size: \t\t{avg_dm_size} packets")
    print(f"Total Elapsed Time: \t\t\t{end-start} sec") 



if __name__ == "__main__":
    main()