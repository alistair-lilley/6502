LDA #$09
JSR fib
BRK

fib:
case0:
CMP #0
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
PHA ; c = # - 1 ; $103
PHA ; b = # - 1 ; $102
SEC
SBC #1
PHA ; a = # - 2 ; $101
JSR fib
TSX
STA $101,X

LDA $102,X
JSR fib
TSX
CLC
ADC $101,X ; a + b
STA $103,X ; = c

PLA
PLA
PLA
RTS