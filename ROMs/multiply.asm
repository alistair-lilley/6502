LDA number
LDY multiplier
JSR multiply
BRK


multiply:
PHA
TSX
DEY

multiplyloop:
CLC
ADC $101,X
DEY
BNE multiplyloop

STA $101,X
PLA
RTS

section .data

number: .byte #5
multiplier: .byte #10