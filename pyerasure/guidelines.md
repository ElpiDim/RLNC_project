# SlidingWindowPyRLNC: Installation & General Usage

1. Clone this repository.
2. At the main directory, build the module with:
      ```
      python3 -m pip install .
      ```

3. Run the ``pyerasure`` examples:
      ```
      python3 examples/path_to_filename
      ```

      Where:
      
      ``path_to_filename`` refers to the path where an example ``.py`` script is located. Currently, there are two subfolders, the ``block`` one, containing example scripts using the block-based implementation, and the ``sw`` folder for testing the sliding window approach.
      

4.  After every change in the module's core files (located in ``src/`` directory), run step 2. again.


Supported BB Version
--------------------
In  folder ``src/pyerasure/block``, one can find the existing block-based RLNC implementation. We provide a test file for this version in folder ``examples/block/`` titled ``block_rlnc_example.py``. It supports the transmission of a huge number of packets divided into smaller blocks. To run this file, one needs to execute the following command:

```
python3 examples/block/block_rlnc_example.py --num_symbols=[NUM_SYMBOLS] --symbol_size=[SYMBOL_SIZE] --block_size=[BLOCK_SIZE] --code_rate=[CODE_RATE] --seed=[SEED] --ploss=[PLOSS]
```

Where:

1. __num_symbols__: Total number of transmitted packets
2. __symbol_size__: Packet size in bytes
3. __block_size__: Block size for encoding (g)
4. __code_rate__: The code rate (R) given in fraction form ``k/n``, where ``k`` describes the mumber of source packets before sending repair packets and ``n`` the sum of source and repair packets transmitted per coding cycle.
5. __seed__: Seed value to control randomness.
6. __ploss__: Channel's packet loss probability given as a float in range [0,100].

There are also some other simpler test files, which demonstrate a simple usage of block-based RLNC either point-to-point or using recoding at an intermediate node.


SW Version with Decoding Matrix Management
--------------------

This version is a functional sliding window RLNC implementation as it deals with practical issues of the sliding window on both sides in terms of computations but there is no consideration on memory overhead (it follows the original PyErasure's block-based design to store data). The encoder uses ``W = d x k`` source packets over a sliding window to generate repair packets and the decoder maintains a sliding-window based decoding matrix to perform decoding operations. The innovation in this version can be detected on the way we handle the decoding matrix size. We introduce the parameter __decoding_depth (d2)__, which defines the _decoding window size_. Similar to the encoding window, where ``CW = d x k``, we define the decoding depth as the integer multiple of k packet groups included in the decoding matrix. This parameter is an integer and in general, it should be at least equal to coding depth value, i.e., ``d2 >= d``. 

We also provide two test files which demonstrate the performance of SW RLNC. File ``examples/sw/sw_rlnc_ge_channel.py`` simulates data transmission on top of a Gilbert Elliott channel whereas ``examples/sw/sw_rlnc_uniform_channel.py`` evaluates performance on top of an i.i.d. channel. These files can be executed given the following command: 

```
python3 examples/sw/filename --num_packets=[NUM_PACKETS] --packet_size=[PACKET_SIZE] --coding_depth=[CODING_DEPTH] --code_rate=[CODE_RATE] --decoding_depth=[DECODING_DEPTH] --seed=[SEED] --ploss=[PLOSS] 
```

Besides code_rate and coding_depth, the rest arguments are optional. For GE channels, there are some additional optional arguments:

```
--burst_duration=[BURST_DURATION] --pe_g=[PE_G] --pe_b=[PE_B]
```

Where:

1. __num_packets__: Total number of transmitted packets
2. __packet_size__: Packet size in bytes
3. __coding_depth__: Coding depth (d)
4. __code_rate__: The code rate (R) given in fraction form ``k/n``, where ``k`` describes the mumber of source packets before sending repair packets and ``n`` the sum of source and repair packets transmitted per coding cycle.
5. __decoding_depth__: The parameter tuning the decoding window size (d2). This parameter should be at least equal to d. 
6. __seed__: Seed value to control randomness.
7. __ploss__: The average channel loss probability (in range [0,1]).
8. __burst_duration__: Average burst duration (in the BAD state) expressed in packets (integer).
9. __pe_g__: The packet loss probability in GOOD state (in range [0,1]).
10. __pe_b__: The packet loss probability in BAD state (in range [0,1]).