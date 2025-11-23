#!/usr/bin/env python
# encoding: utf-8

import os
import random
import argparse
import math
import time

import pyerasure.block as pyblock
import pyerasure.finite_field
import pyerasure.block.generator as pygenerator


def main():
    """
    Simple example showing how to encode and decode blocks of memory using an
    RLNC code.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_symbols", help="Total number of transmitted symbols")
    parser.add_argument("--symbol_size", help="Symbol size in bytes")
    parser.add_argument("--seed", help="Seed to control randomness")
    parser.add_argument("--ploss", help="Channel's packet loss probability")
    parser.add_argument("--block_size", help="Block (generation) size")
    parser.add_argument("--code_rate", help="Ratio of source to total transmitted packets in form k/n")

    args = parser.parse_args()

    symbols = int(args.num_symbols)
    symbol_bytes = int(args.symbol_size)
    seed = int(args.seed)
    loss_probability = float(args.ploss)
    block_size = int(args.block_size)
    try:
        source_k, source_plus_repair = map(int, args.code_rate.split('/'))
        if source_plus_repair - source_k < 0:
            raise ValueError("Code Rate should be in range [0,1]. Please insert appropriate values.")
        repair_n = source_plus_repair - source_k
    except ValueError:
        raise ValueError("Wrong format for code rate.")

    print(f"PyErasure Simulation\n---------------------------------------")
    print(f"Number of Transmitted Symbols: \t{args.num_symbols}")
    print(f"Symbol Size (bytes): \t\t{args.symbol_size}")
    print(f"Block size: \t\t\t{args.block_size}")
    print(f"Loss Probability (%): \t\t{args.ploss}")
    print(f"Seed: \t\t\t\t{args.seed}")

    # Pick the finite field to use for the encoding and decoding.
    field = pyerasure.finite_field.Binary8()

    # Allocate some data to encode - fill data_in with random data
    random.seed(seed)
    data_in = bytearray(random.getrandbits(8) for _ in range(symbols * symbol_bytes))

    elapsed_timeslots = 0  # how many timeslots (transmissions) have elapsed
    transmitted_packets = 0  # counter for the total number of source packets sent from the encoder

    offset = 0
    total_coded = 0
    is_seed_set = False  # to tune generator's seed

    # Per-source-packet metrics
    total_source_packets = symbols
    send_slots = [None] * total_source_packets  #se poio timeslot stalthike 1h fora kathe source packet
    deliver_slots = [None] * total_source_packets  # σε ποιο timeslot egine delivered (decoded) ena paketo
    delivered_packets = 0  # posa source packets paradothikan

    global_symbol_index = 0  # index tou prwtou simvolou tou block

    start = time.time()
    while True:
        if offset >= len(data_in):
            break
        # maybe use memoryview instead, check if error occurs with indexing


        block_data = data_in[offset:offset + block_size * symbol_bytes]
        offset += block_size * symbol_bytes

        # block size (to teleutaio mporei na einai mikrotero)
        current_block_size = symbols % block_size if len(block_data) != block_size * symbol_bytes else block_size

        # Global indices twn symbolwn tou block
        block_first_symbol = global_symbol_index #1o
      #  block_last_symbol = global_symbol_index + current_block_size - 1 #teletaio

        #Create encoder and decoder to handle block size data each time (smaller last block size in case of inexact division)
        encoder = pyblock.Encoder(field, current_block_size, symbol_bytes)
        decoder = pyblock.Decoder(field, current_block_size, symbol_bytes)

        ## Create  generator. The generator must similarly be created
        # based on the encoder/decoder.
        generator = pygenerator.RandomUniform(field, encoder.symbols)
        if not is_seed_set:
            generator.set_seed(seed)
            is_seed_set = True

        encoder.set_symbols(block_data)

        # Number of coded packets to transmit at the end of the block
        coded_packets_per_block = math.ceil(current_block_size / source_k) * repair_n # will result in more redundancy than SW approach if g is not divided by k
        total_coded += coded_packets_per_block

        #se poio timeslot eginei h apokwdikopoihsh tou block
        block_completion_slot = None
#flag gia xameno systematic - an xathei systematic ta ypoloipa den mporoun ana paradothoun amesa in order.
        gap_started  = False

        # Systematic packets
        for index in range(encoder.rank):
            symbol = encoder.symbol_data(index)
            transmitted_packets += 1
            elapsed_timeslots += 1

            global_index = block_first_symbol + index

            # prwth fora p feugei to source
            if send_slots[global_index] is None:
                send_slots[global_index] = elapsed_timeslots


            if random.uniform(0, 100) >= loss_probability:
                decoder.decode_systematic_symbol(symbol, index)

                #an den exei xathei proigoumeno packet tote mporei auto n aparadothei amesa delay = 0
                if not gap_started and deliver_slots[global_index] is None:
                    deliver_slots[global_index] = elapsed_timeslots
                    delivered_packets += 1

                #check gt Μporei na oloklhrwthike mono me systematic (xwris kan na steiloume coded) epeidh de xathike kanena
                if decoder.is_complete() and block_completion_slot is None:
                    block_completion_slot = elapsed_timeslots
            else:
                gap_started =True

        # Coded packets
        for r in range(coded_packets_per_block):
            coefficients = generator.generate()
            symbol = encoder.encode_symbol(coefficients)
            elapsed_timeslots += 1

            if random.uniform(0, 100) >= loss_probability:
                decoder.decode_symbol(symbol, bytearray(coefficients))

                # check an me auto to coded oloklhrwthei to decode tou block
                if decoder.is_complete() and block_completion_slot is None:
                    block_completion_slot = elapsed_timeslots

        # an ena block ginetai decode leme ta symbols tou einai delivered
        if block_completion_slot is not None:
            for local_index in range(current_block_size):
                global_index = block_first_symbol + local_index
                if deliver_slots[global_index] is None:
                    deliver_slots[global_index] = block_completion_slot
                    delivered_packets += 1

        # next block
        global_symbol_index += current_block_size

    end = time.time()

    print("#####-----Statistics-----#####")
    print(f"Total Transmitted packets: \t{transmitted_packets}")
    print(f"Total coded packets: \t\t{total_coded}")
    print(f"Elapsed timeslots: \t\t{elapsed_timeslots}")
    print(f"Total elapsed time: \t\t{end - start} sec")

    # 1. Delivery ratio
    if total_source_packets > 0:
        delivery_ratio = delivered_packets / total_source_packets
    else:
        delivery_ratio = 0.0
    print(f"Delivery ratio (source delivered / source sent): \t{delivery_ratio:.3f}")

    #2. en seira kathusterhsh
    delays = []
    for i in range(total_source_packets):
        if send_slots[i] is not None and deliver_slots[i] is not None:
            delays.append(deliver_slots[i] - send_slots[i])
    if len(delays)>0:
        avg_delay = sum(delays) / len(delays)
    else:
        avg_delay = 0.0

    print(f"Average in-order delivery delay per source pacjet (slots): \t{avg_delay:.3f}")
if __name__ == "__main__":
    main()
