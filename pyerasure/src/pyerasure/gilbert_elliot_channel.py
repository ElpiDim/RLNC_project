import random
import argparse
from enum import Enum

class GilbertElliotChannel:

    class ChannelState(Enum):
        GOOD = 1
        BAD = 0

    def __init__(self, prob_g2b, prob_b2g, pe_g, pe_b):
        self.prob_g2b = prob_g2b
        self.prob_b2g = prob_b2g
        self.pe_g = pe_g
        self.pe_b = pe_b
        self.current_state = GilbertElliotChannel.ChannelState.GOOD  # Assume starting in good state

    def occurred_GE_erasure(self) -> bool:
        """
        Function updating the state of the channel and deciding if the transmitted packet is lost.

        :return True if the packet was erased.
        """
        # Determine channel state
        if self.current_state == GilbertElliotChannel.ChannelState.GOOD:
            if random.uniform(0, 1) < self.prob_g2b:
                self.current_state = GilbertElliotChannel.ChannelState.BAD
        else:
            if random.uniform(0, 1) < self.prob_b2g:
                self.current_state = GilbertElliotChannel.ChannelState.GOOD

        # Check if erasure occurs
        if self.current_state == GilbertElliotChannel.ChannelState.GOOD:
            return random.uniform(0, 1) < self.pe_g
        else:
            return random.uniform(0, 1) < self.pe_b


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prob_g2b", help="Transition probability from Good to Bad state")
    parser.add_argument("--prob_b2g", help="Transition probability from Bad to Good state")
    parser.add_argument("--pe_g", help="Packet erasure probability in Good state")
    parser.add_argument("--pe_b", help="Packet erasure probability in Bad state")
    parser.add_argument("--seed", help="Seed to control randomness")

    args = parser.parse_args()

    prob_g2b = float(args.prob_g2b)
    prob_b2g = float(args.prob_b2g)
    pe_g = float(args.pe_g)
    pe_b = float(args.pe_b)
    seed = int(args.seed)
    random.seed(seed)

    channel = GilbertElliotChannel(prob_g2b, prob_b2g, pe_g, pe_b)

    # Simulate transmission of 100 packets
    for i in range(100):
        if channel.occurred_GE_erasure():
            print(f"Packet {i+1} lost")
        else:
            print(f"Packet {i+1} received")
        

if __name__ == "__main__":
    main()
