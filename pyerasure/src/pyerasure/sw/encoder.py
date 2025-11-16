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

from typing import Union
from pyerasure import finite_field


class Encoder:
    """The encoder class is used to encode a set of packets using a "sliding" window."""

    def __init__(
        self,
        field: Union[finite_field.Binary, finite_field.Binary4, finite_field.Binary8],
        packets: int,
        packet_size_bytes: int,
        coding_depth: int,
        source_per_cycle: int,
        repair_per_cycle: int,
    ):
        """
        The encoder constructor.

        :param field: the chosen finite field.
        :param packets: The number of total packets to transmit.
        :param packet_size_bytes: The size of a packet in bytes.
        :param coding_depth: The number of source packet groups used in generating a coded packet.
        :param source_per_cycle: The number of source packets transmitted before redundant ones.
        :param repair_per_cycle: The number of repair packets sent per coding cycle.
        """
        
        self._field = field
        self._packets = packets
        self._packet_size_bytes = packet_size_bytes
        self._packets_data = [None] * packets #buffer to store original data
        self._rank = 0

        self._coding_depth = coding_depth
        self._source_per_cycle = source_per_cycle
        self._repair_per_cycle = repair_per_cycle
        self._coding_window_size = self._coding_depth * self._source_per_cycle
        self._coding_window_start = -1 #lower bound of encoding window
        self._coding_window_end = -1 #points to index up to which (not incl.) encoding window covers
    
    # Attribute getters & setters
    
    @property
    def field(
        self,
    ) -> Union[finite_field.Binary, finite_field.Binary4, finite_field.Binary8]:
        """The chosen finite field."""
        return self._field
        
    @property
    def packets(self) -> int:
        """The number of packets."""
        return self._packets

    @property
    def packet_size_bytes(self) -> int:
        """The size of a packet in bytes."""
        return self._packet_size_bytes

    @property
    def rank(self) -> int:
        """The total number of stored source packets."""
        return self._rank

    @property
    def coding_depth(self) -> int:
        return self._coding_depth
    
    @property
    def source_per_cycle(self) -> int:
        return self._source_per_cycle
    
    @property
    def repair_per_cycle(self) -> int:
        return self._repair_per_cycle
    
    @property
    def coding_window_size(self) -> int:
        return self._coding_window_size
    
    @property
    def coding_window_start(self) -> int:
        return self._coding_window_start
    
    @coding_window_start.setter
    def coding_window_start(self, value: int):
        self._coding_window_start = value
    
    @property
    def coding_window_end(self) -> int:
        return self._coding_window_end
    
    @coding_window_end.setter
    def coding_window_end(self, value: int):
        self._coding_window_end = value     

    #This function is only used when we want to load all original data on the buffer at once
    def set_packets(self, block_data: bytes):
        """
        Set all packets.

        :param block_data: The data of the block.
        """
        if len(block_data) != self.packets * self.packet_size_bytes: #the expected size of the whole block in bytes
            raise ValueError(f"Invalid block size {block_data}.")
        for index in range(self.packets):
            offset = index * self.packet_size_bytes
            self.set_packet(
                index,
                block_data[offset : offset + self.packet_size_bytes],
            )
        
    def set_packet(self, index: int, packet_data: bytes):
        """
        Set a single packet.

        :param index: The index of the packet.
        :param packet_data: The data of the packet.
        """
        if len(packet_data) != self.packet_size_bytes:
            raise ValueError(f"Invalid packet size {len(packet_data)}.")
        if index >= self.packets:
            raise ValueError(f"Invalid packet index. {index}")
        if index != self.rank:
            raise ValueError("Packets must be set in order.")
        self._packets_data[index] = packet_data
        self._rank += 1  

    def window_len(self) -> int:
        """
        Calculates the length of the encoding window.

        :return int
        """
        return self.coding_window_end - self.coding_window_start

    def is_window_full(self) -> bool:
        """
        Checks whether the coding window contains |coding_window_size| elements.

        :return bool
        """
        return self.window_len() == self.coding_window_size
    
    def is_window_empty(self) -> bool:
        """
        Checks whether the coding window is empty.

        :return bool
        """
        return self.coding_window_end == self.coding_window_start

    def update_coding_window(self, index: int):
        """
        Function to update the bounds of the coding window. When it gets full, the window should slide.

        :param index: The id of the transmitted packet.
        """      
        if self.is_window_empty():
            self.coding_window_start = index
            self.coding_window_end = index + 1
            #print(f"update_coding_window(): in is_empty() - New values: Start: {self.coding_window_start} - End: {self.coding_window_end} - Max CW Size: {self.coding_window_size}.")
        elif self.window_len() < self.coding_window_size:
            self.coding_window_end += 1
            #print(f"update_coding_window(): not full not empty - New values: Start: {self.coding_window_start} - End: {self.coding_window_end} - Max CW Size: {self.coding_window_size}.")
        elif self.is_window_full(): 
            self.coding_window_start += 1
            self.coding_window_end += 1
            #print(f"update_coding_window(): in is_full() - New values: Start: {self.coding_window_start} - End: {self.coding_window_end} - Max CW Size: {self.coding_window_size}.")          


    def is_packet_set(self, index: int) -> bool:
        """
        Check if a packet is set.

        :param index: The index of the packet.
        :return: True if the packet set.
        """
        if index >= self.packets:
            raise ValueError("Invalid packet index.")
        return self._packets_data[index] is not None

    def packet_data(self, index: int) -> bytes:
        """
        Get the data (payload) of a packet.

        :param index: The index of the packet.
        :return: The data of the packet.
        """
        if index >= self.packets:
            raise ValueError("Invalid packet index.")
        return self._packets_data[index]
    
    def calculate_num_coefficients(self) -> int:
        """
        Computes the number of coefficients for a repair packet. 
        If the window is not full, it uses the number of the current contents of the window.

        :return: the number of coefficients to be generated.
        """
        if not self.is_window_full():
            return self.window_len()
        return self.coding_window_size

    def encode_packet(self, coefficients: bytes) -> bytearray:
        """
        Encode a packet based on the given coefficients.

        :param coefficients: The coding coefficients that describe the
                             encoding.
        :return: The encoded packet.
        """
        encoded_packet = bytearray(self.packet_size_bytes)

        #print(f"encode_packet(): CW start: {self.coding_window_start} CW end: {self.coding_window_end} -- length of coefficients: {len(coefficients)}.")
        for index in range(self.rank):
            coefficient = self.field.get_value(coefficients, index)
            if coefficient == 0:
                continue

            if not self.is_packet_set(index):
                raise ValueError(f"Packet with index {index} not set.")
            self.field.vector_multiply_add_into(
                encoded_packet, self.packet_data(index), coefficient
            )

        return encoded_packet