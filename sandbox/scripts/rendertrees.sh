#!/bin/bash
CMD=neato
CMD=twopi

CMD=circo
CMD=dot
CMD=fdp

./rendertrees.py -pExpr -s0 -m1          | $CMD -Tpdf -o figures/exprsub.pdf
./rendertrees.py -pTerminal -s0 -m1      | $CMD -Tpdf -o figures/terminalsub.pdf
./rendertrees.py -pFormArgument -s0 -m1  | $CMD -Tpdf -o figures/formargument.pdf
./rendertrees.py -pOperator -s0 -m1      | $CMD -Tpdf -o figures/operatorsub.pdf

./rendertrees.py -pExpr -s0              | $CMD -Tpdf -o figures/expr.pdf
./rendertrees.py -pTerminal -s0          | $CMD -Tpdf -o figures/terminal.pdf
./rendertrees.py -pOperator -s0          | $CMD -Tpdf -o figures/operator.pdf

./rendertrees.py -pFunction -s0          | $CMD -Tpdf -o figures/function.pdf
./rendertrees.py -pGeometricQuantity -s0 | $CMD -Tpdf -o figures/geometricquantity.pdf
./rendertrees.py -pConstantValue -s0     | $CMD -Tpdf -o figures/constantvalue.pdf
./rendertrees.py -pUtilityType -s0       | $CMD -Tpdf -o figures/utilitytype.pdf

./rendertrees.py -pAlgebraOperator -s0   | $CMD -Tpdf -o figures/algebraoperator.pdf
./rendertrees.py -pCondition -s0         | $CMD -Tpdf -o figures/condition.pdf
./rendertrees.py -pMathFunction -s0      | $CMD -Tpdf -o figures/mathfunction.pdf
./rendertrees.py -pWrapperType -s0       | $CMD -Tpdf -o figures/wrappertype.pdf
./rendertrees.py -pRestricted -s0        | $CMD -Tpdf -o figures/restricted.pdf
./rendertrees.py -pDerivative -s0        | $CMD -Tpdf -o figures/derivative.pdf

./rendertrees.py -pCompoundDerivative -s0     | $CMD -Tpdf -o figures/compoundderivative.pdf
./rendertrees.py -pCompoundTensorOperator -s0 | $CMD -Tpdf -o figures/compoundtensoroperator.pdf

