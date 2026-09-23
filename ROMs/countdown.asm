LDX number
JSR decrement
BRK

decrement:
DEX
BNE *-1
RTS

number: .byte #$05