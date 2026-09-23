LDX #$09
JSR fib
BRK



fib:
TXA

case0:
BNE case1
LDA #0
RTS

case1:
SEC
SBC #1
BNE casen
LDA #1
RTS

casen:
TAY
LDA #1
PHA
LDA #0
PHA
LDA #0
PHA
TSX

fibloop:
LDA $103,X
STA $101,X
CLC
ADC $102,X
STA $103,X
LDA $101,X
STA $102,X
DEY
BEQ endloop
JMP fibloop

endloop:
PLA
PLA
PLA
RTS