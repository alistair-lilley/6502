from __future__ import annotations

import time

import numpy as np

from typing import Dict, Callable, Tuple
from enum import Enum


class AddressMode(Enum):
    Accumulator = 0
    Immediate = 1
    ZeroPage = 2
    ZeroPageX = 3
    ZeroPageY = 4
    Relative = 5
    Absolute = 6
    AbsoluteX = 7
    AbsoluteY = 8
    Indirect = 9
    IndirectX = 10
    IndirectY = 11

class ProcessStatusRegister:
    """ Process status/flags registers """

    def __init__(self: ProcessStatusRegister) -> None:
        self.carry: bool = False # C
        self.zero: bool = False # Z
        self.interrupt_disable: bool = False # I
        self.decimal_mode: bool = False # D
        self.break_command: bool = False # B
        self.fifth_bit: bool = True # 1
        self.overflow: bool = False # V
        self.negative: bool = False # N

        self.register = np.uint8(0)

    def compile_flags(self: ProcessStatusRegister) -> None:
        for bitpos, flag in enumerate((
            self.carry,
            self.zero,
            self.interrupt_disable,
            self.decimal_mode,
            self.break_command,
            self.fifth_bit,
            self.overflow,
            self.negative
        )):
            self.register |= ((flag & 0b1) << bitpos)

    def expand_flags(self: ProcessStatusRegister) -> None:
        self.carry = bool(self.register & 0b1)
        self.zero = bool((self.register >> 1) & 0b1)
        self.interrupt_disable = bool((self.register >> 2) & 0b1)
        self.decimal_mode = bool((self.register >> 3) & 0b1)
        self.break_command = bool((self.register >> 4) & 0b1)
        self.fifth_bit = bool((self.register >> 5) & 0b1)
        self.overflow = bool((self.register >> 6) & 0b1)
        self.negative = bool((self.register >> 7) & 0b1)


class RAM:
    """ RAM mapping 
    $0000-$00FF -- Zero page
    $0100-$01FF -- Stack
    $0200-$FFFA -- FREE
    $FFFA,$FFFB -- non-maskable interrupt handler
    $FFFC,$FFFD -- power on reset location
    $FFFE,$FFFF -- BRK/interrupt handler
    """

    def __init__(self: RAM) -> None:
        self.ram = np.zeros(0x10000, dtype=np.uint8)


class CPU:
    """ CPU """

    def __init__(self: CPU) -> None:
        self.A = np.uint8(0)
        self.X = np.uint8(0)
        self.Y = np.uint8(0)
        self.S = np.uint8(0)
        self.P = ProcessStatusRegister()
        self.IR = np.uint32(0)
        self.PC = np.uint16(0x0800)
        self.RAM = RAM()

        self.opcode_map: Dict[int, Callable] = {
            # 00-0F
            0x00: ,
            0x01: ,
            0x05: ,
            0x06: ,
            0x08: ,
            0x09: ,
            0x0A: ,
            0x0D: ,
            0x0E: ,
            # 10-1F
            0x10: ,
            0x11: ,
            0x15: ,
            0x16: ,
            0x18: ,
            0x19: ,
            0x1D: ,
            0x1E: ,
            # 20-2F
            0x20: ,
            0x21: ,
            0x24: ,
            0x25: ,
            0x26: ,
            0x28: ,
            0x29: ,
            0x2A: ,
            0x2C: ,
            0x2D: ,
            0x2E: ,
            # 30-3F
            0x30: ,
            0x31: ,
            0x35: ,
            0x36: ,
            0x38: ,
            0x39: ,
            0x3D: ,
            0x3E: ,
            # 40-4F
            0x40: ,
            0x41: ,
            0x45: ,
            0x46: ,
            0x48: ,
            0x49: ,
            0x4A: ,
            0x4C: ,
            0x4D: ,
            0x4E: ,
            # 50-5F
            0x50: ,
            0x51: ,
            0x55: ,
            0x56: ,
            0x58: ,
            0x59: ,
            0x5D: ,
            0x5E: ,
            # 60-6F
            0x60: ,
            0x61: ,
            0x65: ,
            0x66: ,
            0x68: ,
            0x69: ,
            0x6A: ,
            0x6C: ,
            0x6D: ,
            0x6E: ,
            # 70-7F
            0x70: ,
            0x71: ,
            0x75: ,
            0x76: ,
            0x78: ,
            0x79: ,
            0x7D: ,
            0x7E: ,
            # 80-8F
            0x81: ,
            0x84: ,
            0x85: ,
            0x86: ,
            0x88: ,
            0x8A: ,
            0x8C: ,
            0x8D: ,
            0x8E: ,
            # 90-9F
            0x90: ,
            0x91: ,
            0x94: ,
            0x95: ,
            0x96: ,
            0x98: ,
            0x99: ,
            0x9A: ,
            0x9D: ,
            # A0-AF
            0xA0: ,
            0xA1: ,
            0xA2: ,
            0xA4: ,
            0xA5: ,
            0xA6: ,
            0xA8: ,
            0xA9: ,
            0xAA: ,
            0xAC: ,
            0xAD: ,
            0xAE: ,
            # B0-BF
            0xB0: ,
            0xB1: ,
            0xB4: ,
            0xB5: ,
            0xB6: ,
            0xB8: ,
            0xB9: ,
            0xBA: ,
            0xBC: ,
            0xBD: ,
            0xBE: ,
            # C0-CF
            0xC0: ,
            0xC1: ,
            0xC4: ,
            0xC5: ,
            0xC6: ,
            0xC8: ,
            0xC9: ,
            0xCA: ,
            0xCC: ,
            0xCD: ,
            0xCE: ,
            # D0-DF
            0xD0: ,
            0xD1: ,
            0xD5: ,
            0xD6: ,
            0xD8: ,
            0xD9: ,
            0xDD: ,
            0xDE: ,
            # E0-EF
            0xE0: ,
            0xE1: ,
            0xE4: ,
            0xE5: ,
            0xE6: ,
            0xE8: ,
            0xE9: ,
            0xEA: ,
            0xEC: ,
            0xED: ,
            0xEE: ,
            # F0-FF
            0xF0: ,
            0xF1: ,
            0xF5: ,
            0xF6: ,
            0xF8: ,
            0xF9: ,
            0xFD: ,
            0xFE: ,
}

    def _cycle(self: CPU, count: int) -> None:
        for _ in range(count):
            time.sleep(1*10^-6)

    def fetch_decode_execute(self: CPU) -> None:
        opcode: int = int((self.IR >> 24) & 0xFF) # 8-bit int
        self.opcode_map[opcode]()

    def _get_args(self: CPU) -> Tuple[np.uint8, np.uint16]:
        arg1: np.uint8 = np.uint8((self.IR >> 16) & 0xFF)
        arg2: np.uint8 = np.uint8((self.IR >> 8) & 0xFF)
        eightbitarg = arg2
        sixteenbitarg = np.uint16(np.uint16(arg1) << 8 | np.uint16(arg2))
        return eightbitarg, sixteenbitarg

    def _adc(self: CPU, address_mode: AddressMode) -> None:
        _8bitarg, _16bitarg = self._get_args()
        oldA = self.A
        cycles = 0
        match address_mode:
            case AddressMode.Immediate:
                self.A += _8bitarg
                cycles = 2
            case AddressMode.ZeroPage:
                self.A += self.RAM.ram[_8bitarg]
                cycles = 3
            case AddressMode.ZeroPageX:
                self.A += self.RAM.ram[_8bitarg + self.X]
                cycles = 4
            case AddressMode.ZeroPageY:
                self.A += self.RAM.ram[_8bitarg + self.Y]
                cycles = 4
            case AddressMode.Absolute:
                self.A += self.RAM.ram[_16bitarg]
                cycles = 4
            case AddressMode.AbsoluteX:
                self.A += self.RAM.ram[_16bitarg + self.X]
                cycles = 4
            case AddressMode.AbsoluteY:
                self.A += self.RAM.ram[_16bitarg + self.Y]
                cycles = 4
            # NOTE: Fix later, these ones are weird
            # case AddressMode.IndirectX:
            #     self.A += self.RAM.ram[self.RAM.ram[eightbitarg + self.X]]
            # case AddressMode.IndrectY:
            #     self.A += self.RAM.ram[self.RAM.ram]
        if oldA > self.A:
            self.P.carry = True
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        # NOTE: Figure out how to write case for setting overflow
        self._cycle(cycles)
    
    def _and(self: CPU, address_mode: AddressMode) -> None:
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        match address_mode:
            case AddressMode.Immediate:
                cycles = 2
                self.A &= _8bitarg
            case AddressMode.ZeroPage:
                cycles = 3
                self.A &= self.RAM.ram[_8bitarg]
            case AddressMode.ZeroPageX:
                cycles = 4
                self.A &= self.RAM.ram[_8bitarg + self.X]
            case AddressMode.ZeroPageY:
                cycles = 4
                self.A &= self.RAM.ram[_8bitarg + self.Y]
            case AddressMode.Absolute:
                cycles = 4
                self.A &= self.RAM.ram[_16bitarg]
            case AddressMode.AbsoluteX:
                cycles = 4
                self.A &= self.RAM.ram[_16bitarg + self.X]
            case AddressMode.AbsoluteY:
                cycles = 4
                self.A &= self.RAM.ram[_16bitarg + self.Y]
            # NOTE: Fix later, these ones are weird
            # case AddressMode.IndirectX:
            #     self.A += self.RAM.ram[self.RAM.ram[eightbitarg + self.X]]
            # case AddressMode.IndrectY:
            #     self.A += self.RAM.ram[self.RAM.ram]
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        self._cycle(cycles)

    def _asl(self: CPU, address_mode: AddressMode) -> None:
        _8bitarg, _16bitarg = self._get_args()
        cycles = 0
        match address_mode:
            case AddressMode.Accumulator:
                self.P.carry = bool((self.A >> 7) & 0b1)
                self.A <<= 1
                cycles = 2
            case AddressMode.ZeroPage:
                self.P.carry = bool((self.RAM.ram[_8bitarg] >> 7) & 0b1)
                self.RAM.ram[_8bitarg] <<= 1
                cycles = 5
            case AddressMode.ZeroPageX:
                self.P.carry = bool((self.RAM.ram[_8bitarg + self.X] >> 7) & 0b1)
                self.RAM.ram[_8bitarg + self.X] <<= 1
                cycles = 6
            case AddressMode.Absolute:
                self.P.carry = bool((self.RAM.ram[_16bitarg] >> 7) & 0b1)
                self.RAM.ram[_16bitarg] <<= 1
                cycles = 6
            case AddressMode.AbsoluteX:
                self.P.carry = bool((self.RAM.ram[_16bitarg + self.X] >> 7) & 0b1)
                self.RAM.ram[_16bitarg + self.X] <<= 1
                cycles = 7
        if self.A == 0:
            self.P.zero = True
        if (self.A >> 7) & 0b1 == 1:
            self.P.negative = True
        self._cycle(cycles)