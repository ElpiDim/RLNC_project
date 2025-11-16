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

from typing import Tuple, Union
from enum import Enum
import math

from pyerasure import finite_field


class Decoder:
    """
    The Decoder class is an optimized version of a SW RLNC decoder.
    Apart from removing DECODED packets, we also remove MISSING packets from the decoding window 
    if they fall beyond the active CW, as no further coded packets will contain information for them in the future. 
    However, regarding the PARTIALLY_DECODED packets, we can remove them earlier only if they rely on a MISSING packet 
    that can be safely removed. However, the general rule is that they stay as long as max_index – min_index < decoding window size.
    """

    class PacketStatus(Enum):
        MISSING = 0
        PARTIALLY_DECODED = 1
        DECODED = 2

    def __init__(
        self,
        field: Union[finite_field.Binary, finite_field.Binary4, finite_field.Binary8],
        packets: int,
        packet_size_bytes: int,
        coding_depth: int,
        source_per_cycle: int,
        decoding_depth: int,
    ):
        """
        The Decoder constructor.

        :param field: the chosen finite field.
        
        :param packets: The number of total packets to transmit.
        :param packet_size_bytes: The size of a packet in bytes.
        :param coding_depth: The number of source packet groups used in generating a coded packet.
        :param source_per_cycle: The number of source packets transmitted before redundant ones.
        :param decoding_depth: Defines the decoding window size, in analogy with coding_depth but generally decoding_depth >= coding_depth.
        """
        self._field = field
        self._packets = packets
        self._packet_size_bytes = packet_size_bytes

        self._rank = 0 
        self._packets_data = [None] * packets #buffer to store original data
        self._coefficients = [None] * packets #buffer to store corresponding coefficients
        self._packet_status = [Decoder.PacketStatus.MISSING] * packets

        self._min_index = 0 #the index of the oldest source packet still existing in the decoding matrix
        self._max_index = -1 #the index of the most recent source packet that has been received or seen (even out-of-order)

        self._coding_window = coding_depth * source_per_cycle #sender's encoding window size
        if decoding_depth < coding_depth:
            raise ValueError(f"Decoding depth should be at least equal to the coding depth ({decoding_depth}, {coding_depth}).")
        self._decoding_window = decoding_depth * source_per_cycle #receiver's decoding window size

        # Statistics
        self.delivered_packets = 0 # total number of correct packets delivered to the receiver (received and fully decoded)
        self.complete_decodings = 0 # total number of complete decodings
        self.dm_size = 0 # used to calculate avg decoding matrix size
        self._last_delivered = -1 # index of the latest source packet delivered in-order to the upper layer
        self.packet_delays = {} # dictionary storing packet index and delivery time
        self.current_timeslot = 0 # helper variable for calculating packet delay, avoid changing decode() signature
        self.sum_of_delay = 0 # used to calculate average packet delay
        self.useful_decoded_packets = 0 # counter of packets that were actually decoded and delivered to the application
        self.packet_operations = 0 # counter of packet operations during FS and BS steps
        self.optimal_non_zero_coefficient_operations = 0 # number of non zero coefficients involved in GF operations (optimal case)
        self.actual_coefficient_operations = 0 # number of coefficient GF operations in current implementation  
        self.decodings_counter = 0 # counter of all decoding attempts of possible innovative packets (may not innovative because of linear dependency)
        

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
        """The rank of the decoding matrix."""
        return self._rank
    
    @property
    def coding_window(self) -> int:
        """The coding window size."""
        return self._coding_window
    
    # Pointers for handling the decoding matrix size

    @property
    def min_index(self) -> int:
        """The index of the oldest packet in the decoding matrix."""
        return self._min_index
    
    @min_index.setter
    def min_index(self, value: int):
        """Setter function to update the value of the pointer min_index."""
        self._min_index = value

    @property
    def max_index(self) -> int:
        """The index of the most recently received packet."""
        return self._max_index
    
    @max_index.setter
    def max_index(self, value: int):
        """Setter function to update the value of the pointer max_index."""
        self._max_index = value
    
    @property
    def decoding_window(self) -> int:
        """The decoding window size."""
        return self._decoding_window
    
    @property
    def last_delivered(self) -> int:
        return self._last_delivered

    @last_delivered.setter
    def last_delivered(self, value: int):
        self._last_delivered = value

    # Remains only for compatibility reasons. It's not used anymore.
    def is_complete(self) -> bool:
        """
        Check if the Decoder is complete.

        :return: True if the Decoder is complete.
        """
        return self._rank == self._packets    

    def is_packet_missing(self, index: int) -> bool:
        """
        Check if a packet is missing.

        :param index: The index of the packet.
        :return: True if the packet is missing.
        """
        
        return self._packet_status[index] == Decoder.PacketStatus.MISSING

    def is_packet_pivot(self, index: int) -> bool:
        """
        Check if a packet is a pivot packet.

        :param index: The index of the packet.
        :return: True if the packet is a pivot packet.
        """
        
        return self._packet_status[index] != Decoder.PacketStatus.MISSING

    def is_packet_decoded(self, index: int) -> bool:
        """
        Check if a packet is decoded.

        :param index: The index of the packet.
        :return: True if the packet is decoded.
        """
         
        if self._packet_status[index] != Decoder.PacketStatus.DECODED:
            # Check coefficients
            if self.__is_coefficients_decoded(index):
                self._packet_status[index] = Decoder.PacketStatus.DECODED
                return True
            return False
        else:
            return True
        
    def __is_coefficients_decoded(self, index: int):
        """
        Check if the coefficients at the given index are decoded.

        :param index: The index of the coefficients.
        :return: True if the coefficients are decoded, False otherwise.
        """
        coefficients = self.coefficients(index)
        if coefficients is None:
            return False

        for i in range(self.min_index, self.max_index + 1):
            if i == index:
                continue
            if self.field.get_value(coefficients, i) != 0:
                return False

        return True

    def packet_data(self, index: int) -> bytearray:
        """
        Get the data of a packet.

        :param index: The index of the packet.
        """
        if index >= self.packets:
            raise ValueError(f"Invalid packet index {index}.")

        return self._packets_data[index]

    # Not practically used
    def block_data(self) -> bytes:
        """
        Get the data of the whole block.

        :return: The data of the block.
        """

        block_data = bytearray()
        for i in range(self.packets):
            packet_data = self.packet_data(i)
            if packet_data is None:
                packet_data = bytearray(self.packet_size_bytes)
            block_data.extend(packet_data)
        return block_data

    def coefficients(self, index: int) -> bytearray:
        """
        Get the coefficients of a packet.

        :param index: The index of the packet.
        """
        if index >= self.packets:
            raise ValueError(f"Invalid packet index {index}")
        
        return self._coefficients[index]


    def update_min_index(self):
        count = 0
        while count <= self.max_index:
            if self.is_packet_missing(self.min_index) or self.is_packet_decoded(self.min_index):
                if self.max_index - self.min_index < self.coding_window:
                    # missing packets without any related information cannot be decoded if their index exceeds the current coding window
                    # decoded packets shall be removed if their index is not in the range of the current coding window
                    return
            else:
                # partially decoded packet
                if self.max_index - self.min_index < self.decoding_window:
                    # partially decoded packets that are in range of the decoding window should be checked for dependencies 
                    # with missing packets. If they rely on missing packets that are outside CW then they can be removed
                    assert self._packet_status[self.min_index] == Decoder.PacketStatus.PARTIALLY_DECODED
                    current_coefficients = self.coefficients(self.min_index)[self.min_index:self.max_index+1]
                    nonzero_indices = [self.min_index + i for i, elem in enumerate(current_coefficients) if elem != 0]
                    nonzero_indices[:2] if len(nonzero_indices) > 2 else nonzero_indices[:]
                    # if second non zero index outside current coding window, remove partially decoded packet
                    if self.max_index - nonzero_indices[-1] < self.coding_window: 
                        return
                    #print(f"Packet {self.min_index} will be removed")
                
            self.min_index += 1
            count += 1

            if self.last_delivered + 1 < self.min_index:
                self.packet_delays[self.last_delivered + 1] = -1
                self.last_delivered += 1
                self.__compute_inorder_delay()


    # helper function - used in decode_packet()
    def __update_packet_status(self):
        for index in range(self.min_index, self.max_index + 1):
            if self._packet_status[index] != Decoder.PacketStatus.DECODED and self.__is_coefficients_decoded(index):
                self._packet_status[index] = Decoder.PacketStatus.DECODED
                self.delivered_packets += 1
                self.useful_decoded_packets += 1

    # function to calculate in-order packet delay
    def __compute_inorder_delay(self):
        count = 0
        while count <= self.max_index:
            if self.last_delivered + 1 >= self.packets:
                #print(f"__compute_inorder_delay(): Index will go out of range")
                return
            if self._packet_status[self.last_delivered + 1] != Decoder.PacketStatus.DECODED:
                return
            if self._packet_status[self.last_delivered + 1] == Decoder.PacketStatus.DECODED:
                packet_delay = self.current_timeslot - self.packet_delays.get(self.last_delivered + 1)
                self.sum_of_delay += packet_delay
                self.packet_delays[self.last_delivered + 1] = packet_delay
                self.last_delivered += 1
                count += 1
    
    # helper function - for average delay of packets remaining in the buffer at the end of transmission
    def sum_delay_of_last(self):
        for index in range(self.last_delivered + 1, self.packets):
            if self._packet_status[index] != Decoder.PacketStatus.DECODED:
                self.packet_delays[index] = -1
            else:
                packet_delay = self.current_timeslot - self.packet_delays.get(index)
                self.sum_of_delay += packet_delay
                self.packet_delays[index] = packet_delay

    # helper function to store the start time of source packets
    def update_timestamps(self, index: int, timeslot: int):
        if index not in self.packet_delays.keys():
            self.packet_delays[index] = timeslot

    
    def decode_packet(self, packet_data: bytearray, coefficients: bytearray, window_bounds: Tuple[int,int]):
        """
        Feed a coded packet to the Decoder.

        :param packet_data: The data of the packet assumed to be packet_size_bytes
                            bytes in size.
        :param coefficients: The coding coefficients that describe the
                            encoding performed on the packet.
        :param window_bounds: Start and end index of the coding window. 
                            Used to update max_index in Decoder.
        """
        

        # if coded packet's upper bound is larger than max_index, there is high probability that the packet is innovative
        # But if the corresponding coefficients equal 0, it is not (at least for indexes after max_index). 
        # If we update max_index, we need to update min_index as well
        # otherwise, the Decoder's extended window will be even larger and there is some probability that the pivot index 
        # is detected in that temporary (but wrong) "extension" of the extended window.
        # Naive approach: check corresponding coefficients value. If not zero, the packet is innovative and indexes should be updated.

        assert window_bounds[1] - window_bounds[0] + 1 <= self._coding_window
        assert self.max_index <= window_bounds[1]
        if self.max_index < window_bounds[1]:
            temp_coeff = coefficients[self.max_index + 1 : window_bounds[1] + 1]
            if not temp_coeff == bytearray(len(temp_coeff)):
                #print("Repair packet is innovative. Update indexes")
                self.max_index = window_bounds[1]
                self.update_min_index()


        # decide if the received packet is linearly dependent - if so discard it without processing to avoid computational overhead
        # check if there is any missing packet in the DW (for which the coded packet can compensate)
        # if the max missing index is not covered by the coded packet, then coded packet is discarded (not useful) and the window is advanced
        losses = []
        [losses.append(index) for index in range(self.min_index, self.max_index + 1) if self._packet_status[index] == Decoder.PacketStatus.MISSING]
        maxloss = max(losses) if len(losses) > 0 else -1
        if (len(losses) == 0 or (len(losses) > 0 and maxloss < window_bounds[0])):
            #print("Coded packet non innovative. Break")
            return
        # A coded packet is also not useful when the maxloss > window_bounds[1]. However, this can happen only if the order of packet arrival is different. 
        # Here, this is not possible as packets are received in the order they were sent (point to point channel).
        
        self.decodings_counter += 1
        
        
        pivot_index = self.__forward_substitute_to_pivot(packet_data, coefficients)   
        if pivot_index is None:
            return
        if not self.field.is_binary():
            self.__normalize(packet_data, coefficients, pivot_index)
        self.__forward_substitute_from_pivot(packet_data, coefficients, pivot_index)
        self.__backward_substitute(packet_data, coefficients, pivot_index)

        # Store coded packet
        self._packets_data[pivot_index] = packet_data
        self._coefficients[pivot_index] = coefficients

        if self.__is_coefficients_decoded(pivot_index):
            self._packet_status[pivot_index] = Decoder.PacketStatus.DECODED
            self.delivered_packets += 1
            self.useful_decoded_packets += 1
        else:
            self._packet_status[pivot_index] = Decoder.PacketStatus.PARTIALLY_DECODED
        
        self._rank += 1
        self.complete_decodings += 1
        self.dm_size += (self.max_index - self.min_index + 1)

        # update packet status (in case previous partially decoded packets become decoded)
        self.__update_packet_status()

        # calculate in order delay
        if self.is_packet_decoded(pivot_index) and pivot_index == self.last_delivered + 1:
            self.last_delivered += 1
            packet_delay = self.current_timeslot - self.packet_delays.get(pivot_index)
            self.sum_of_delay += packet_delay
            self.packet_delays[pivot_index] = packet_delay
        self.__compute_inorder_delay()
        
        # Update pointers
        if pivot_index > self.max_index:
            self.max_index = pivot_index
        self.update_min_index()


    def decode_systematic_packet(self, packet_data: bytearray, index: int):
        """
        Feed a systematic, i.e, un-coded packet to the Decoder.

        :param packet_data: The data of the packet assumed to be packet_size_bytes
        bytes in size.
        :param index: The index of the given packet.
        """

        if index >= self.packets:
            raise ValueError(f"Invalid packet index {index}")
        
        if self.is_packet_decoded(index):
            return

        if self.is_packet_pivot(index):
            self.__swap_decode(packet_data, index)

        # Store the packet
        self._packets_data[index] = packet_data
        self._coefficients[index] = bytearray(
            self.field.elements_to_bytes(self.packets)
        )
        self.field.set_value(self.coefficients(index), index, 1)
        self._packet_status[index] = Decoder.PacketStatus.DECODED

        # increase counter of received packets
        self._rank += 1
        self.delivered_packets += 1

        # calculate in order delay
        if index == self.last_delivered + 1:
            self.last_delivered += 1
            packet_delay = self.current_timeslot - self.packet_delays.get(index)
            self.sum_of_delay += packet_delay
            self.packet_delays[index] = packet_delay
        self.__compute_inorder_delay()

        # Update pointers
        if index > self.max_index:
            self.max_index = index
        # fixes the problem with the interleaved coded packets, but is not useful if the coded packets do not have full size
        #self.__backward_substitute(self._packets_data[index], self._coefficients[index], index)    
        self.update_min_index()


    """
    #TODO: Implement recoding
    def recode_packet(self, coefficients_in: bytes) -> Tuple[bytes, bytearray]:
        #Recodes a new packet based on given the coefficients and current state
        #of the Decoder.

        #:param coefficients_in: These are the coding coefficients.
        #:return: The recoded packet and resulting coefficients.

        packet_data = bytearray(self.packet_size_bytes)
        coefficients = bytearray(self.field.elements_to_bytes(self.packets))
 
        for index in range(self.packets):

            value = self.field.get_value(coefficients_in, index)

            if value == 0:
                continue

            assert self.is_packet_pivot(index)

            self.field.vector_multiply_add_into(
                coefficients, self.coefficients(index), value
            )
            self.field.vector_multiply_add_into(
                packet_data,
                self.packet_data(index),
                value,
            )
        return packet_data, coefficients
    """

    def __forward_substitute_to_pivot(
        self, packet_data: bytearray, coefficients: bytearray
    ) -> int:
        """
        Forward substitute the given packet to the pivot packet.

        :param packet_data: The data of the packet.
        :param coefficients: The coefficients of the packet.
        :return: The index of the pivot packet, or none if no pivot packet
        """

        for index in range(self.min_index, self.max_index + 1):
            coefficient = self.field.get_value(coefficients, index)

            if coefficient == 0:
                continue

            if not self.is_packet_pivot(index):
                return index
            
            self.packet_operations += 1

            # GF operations need to be performed only on active DM
            full_coeff = memoryview(coefficients)
            sliced_coeff = full_coeff[self.min_index : self.max_index + 1]

            full_coeff2 = memoryview(self.coefficients(index))
            sliced_coeff2 = full_coeff2[self.min_index : self.max_index + 1]
            
            non_zero_coeffs = sum(x > 0 for x in list(sliced_coeff2))
            self.optimal_non_zero_coefficient_operations += non_zero_coeffs
            self.actual_coefficient_operations += len(sliced_coeff2)

            self.field.vector_multiply_subtract_into(
                sliced_coeff, sliced_coeff2, coefficient
            )
            # number of iterations depends on packet_size
            self.field.vector_multiply_subtract_into(
                packet_data, self.packet_data(index), coefficient
            )

        return None

    def __forward_substitute_from_pivot(
        self, packet_data: bytearray, coefficients: bytearray, pivot: int
    ):
        """
        Forward substitute the given packet from the pivot packet.

        :param packet_data: The data of the packet.
        :param coefficients: The coefficients of the packet.
        :param pivot: The index of the pivot packet.
        """

        # Start right after the pivot_index position
        for index in range(pivot + 1, self.max_index + 1):
            coefficient = self.field.get_value(coefficients, index)

            if coefficient == 0:
                continue

            if not self.is_packet_pivot(index):
                continue
            
            self.packet_operations += 1

            full_coeff = memoryview(coefficients)
            sliced_coeff = full_coeff[pivot + 1 : self.max_index + 1]

            full_coeff2 = memoryview(self.coefficients(index))
            sliced_coeff2 = full_coeff2[pivot + 1 : self.max_index + 1]

            non_zero_coeffs = sum(x > 0 for x in list(sliced_coeff2))
            self.optimal_non_zero_coefficient_operations += non_zero_coeffs
            self.actual_coefficient_operations += len(sliced_coeff2)
     
            self.field.vector_multiply_subtract_into(
                sliced_coeff, sliced_coeff2, coefficient
            )

            self.field.vector_multiply_subtract_into(
                packet_data, self.packet_data(index), coefficient
            )

    def __backward_substitute(
        self, packet_data: bytearray, coefficients: bytearray, pivot_index: int
    ):
        """
        Backward substitute the given packet.

        :param packet_data: The data of the packet.
        :param coefficients: The coefficients of the packet.
        :param pivot_index: The index of the pivot packet.
        """

        # We found a "1" that nobody else had as pivot, we now
        # subtract this packet from other coded packets
        # - if they have a "1" at our pivot position
        for index in range(self.min_index, self.max_index + 1):

            if index == pivot_index:
                # We cannot backward substitute into our self
                continue
            if self.is_packet_decoded(index):
                # We know that we have no non-zero elements
                # outside the pivot position.
                continue

            if self.is_packet_missing(index):
                # We do not have a packet yet here
                continue

            coefficient = self.field.get_value(self.coefficients(index), pivot_index)

            if coefficient == 0:
                continue
            
            self.packet_operations += 1
            
            full_coeff = memoryview(self.coefficients(index))
            sliced_coeff = full_coeff[self.min_index : self.max_index + 1]

            full_coeff2 = memoryview(coefficients)
            sliced_coeff2 = full_coeff2[self.min_index : self.max_index + 1]

            non_zero_coeffs = sum(x > 0 for x in list(sliced_coeff2))
            self.optimal_non_zero_coefficient_operations += non_zero_coeffs
            self.actual_coefficient_operations += len(sliced_coeff2)

            self.field.vector_multiply_subtract_into(
                sliced_coeff, sliced_coeff2, coefficient
            )

            # Update packet and corresponding vector
            #self.field.vector_multiply_subtract_into(
            #    self.coefficients(index), coefficients, coefficient
            #)
            self.field.vector_multiply_subtract_into(
                self.packet_data(index), packet_data, coefficient
            )

    def __normalize(self, packet_data: bytearray, coefficients: bytearray, index: int):
        """
        Normalize the given packet.

        :param packet_data: The data of the packet.
        :param coefficients: The coefficients of the packet.
        :param index: The index of the packet.
        """
        coefficient = self.field.get_value(coefficients, index)

        inverted_coefficient = self.field.invert(coefficient)
        # Perhaps slicing here will change the result
        self.field.vector_multiply_into(
            coefficients,
            inverted_coefficient,
        )

        self.field.vector_multiply_into(packet_data, inverted_coefficient)

    def __swap_decode(self, packet_data: bytearray, index: int):
        """
        Swap the given packet with an existing coded packet.

        :param packet_data: The data of the packet.
        :param index: The index of the packet.
        """
        # extract packet and coefficients and set the packet as missing
        packet_i = self.packet_data(index)
        coefficients_i = self.coefficients(index)
        self._packet_status[index] = Decoder.PacketStatus.MISSING
        self._rank -= 1

        # Subtract the new pivot packet
        self.field.set_value(coefficients_i, index, 0)
        # Note: add is the same as subtract
        self.field.vector_add_into(packet_i, packet_data)

        # Process the new coded packet: we know that it must
        # contain a larger pivot id than the current (unless it is reduced
        # to all zeroes).
        self.decode_packet(packet_i, coefficients_i)

