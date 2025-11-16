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

import os
import pyerasure.block as pyblock
import pyerasure.finite_field
import pyerasure.block.generator as pygenerator

"""
Full-vector block-based RLNC where the transmitter delivers 15 
packets of size 100 bytes to the receiver.
"""
field = pyerasure.finite_field.Binary8()
encoder = pyblock.Encoder(
    field=field,
    symbols=15,
    symbol_bytes=100)
decoder = pyblock.Decoder(
    field=field,
    symbols=15,
    symbol_bytes=100)

generator = pygenerator.RandomUniform(
    field=field,
    symbols=encoder.symbols)

data_in = bytearray(os.urandom(encoder.block_bytes))
encoder.set_symbols(data_in)

while not decoder.is_complete():
    coefficients = generator.generate()
    symbol = encoder.encode_symbol(coefficients)
    decoder.decode_symbol(symbol, bytearray(coefficients))

assert data_in == decoder.block_data()
print("Success!")