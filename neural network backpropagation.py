# SAG19698726 Aisha Saghir - CMP2020M-2122 Artificial Intelligence – Assignment 1

import math

# architecture = 4, 3, 2, 2
# hidden layer node nets (3)

def training(epochs, x1, x2, x3):

    # input data

    # x0 = [0,0,0,0,0,0]
    y1 = [1,1,1,0,0,0]
    y2 = [0,0,0,1,1,1]

    # weight data
    w04 = 0.9
    w14 = 0.74
    w24 = 0.8
    w34 = 0.35

    w05 = 0.45
    w15 = 0.13
    w25 = 0.4
    w35 = 0.97

    w06 = 0.36
    w16 = 0.68
    w26 = 0.1
    w36 = 0.96

    w07 = 0.98
    w47 = 0.35
    w57 = 0.5
    w67 = 0.9

    w08 = 0.92
    w48 = 0.8
    w58 = 0.13
    w68 = 0.8

    # dw

    dw04 = 0
    dw14 = 0
    dw24 = 0
    dw34 = 0

    dw05 = 0
    dw15 = 0
    dw25 = 0
    dw35 = 0

    dw06 = 0
    dw16 = 0
    dw26 = 0
    dw36 = 0

    dw07 = 0
    dw47 = 0
    dw57 = 0
    dw67 = 0

    dw08 = 0
    dw48 = 0
    dw58 = 0
    dw68 = 0

    iterationno = 0
    testing = epochs + 2

    counter = 0
        
    while testing > 0:
        while epochs > 0 and counter < 3:
            print ('epoch: ', epochs)
            
            # forward propagation
            
            nodenet4 = w04*0 + w14*x1[iterationno] + w24*x2[iterationno] + w34*x3[iterationno] # node a4
            nodenet5 = w05*0 + w15*x1[iterationno] + w25*x2[iterationno] + w35*x3[iterationno] # node a5
            nodenet6 = w06*0 + w16*x1[iterationno] + w26*x2[iterationno] + w36*x3[iterationno] # node a6

            # print (nodenet4, nodenet5, nodenet6)

            # running through sigmoid function
            exponent4 = 1 / (1 + math.exp(-nodenet4))
            exponent5 = 1 / (1 + math.exp(-nodenet5))
            exponent6 = 1 / (1 + math.exp(-nodenet6))

            # print (exponent4, exponent5, exponent6)

            # output layer node nets (2)
            nodenet7 = w07*0 + w47*exponent4 + w57*exponent5 + w67*exponent6 #output1
            nodenet8 = w08*0 + w48*exponent4 + w58*exponent5 + w68*exponent6 #output2

            print ('desired outputs: ', y1[iterationno], y2[iterationno])
            print ('actual outputs: ', nodenet7, nodenet8)

            # back propagation

            error7 = y1[iterationno] - nodenet7
            error8 = y2[iterationno] - nodenet8

            mean = ((error7 + error8) / 2)
            squaremean = mean * mean
            print ('y axis mean^2: ', squaremean)

            # hidden errors 
            error4 = exponent4*(1-exponent4) * ((w47*error7) + (w48*error8))
            error5 = exponent5*(1-exponent5) * ((w57*error7) + (w58*error8))
            error6 = exponent6*(1-exponent6) * ((w67*error7) + (w68*error8))

            # node a4 weight updates
            dw04 = 0.1 * error4 * 0
            dw14 = 0.1 * error4 * x1[iterationno]
            dw24 = 0.1 * error4 * x2[iterationno] 
            dw34 = 0.1 * error4 * x3[iterationno]
            w04 = w04 + dw04
            print ('weight 04: ', w04)
            w14 = w14 + dw14
            print ('weight 14: ', w14)
            w24 = w24 + dw24
            print ('weight 24: ', w24)
            w34 = w34 + dw34
            print ('weight 34: ', w34)

            # node a5 weight updates
            dw05 = 0.1 * error5 * 0
            dw15 = 0.1 * error5 * x1[iterationno]
            dw25 = 0.1 * error5 * x2[iterationno] 
            dw35 = 0.1 * error5 * x3[iterationno]
            w05 = w05 + dw05
            print ('weight 05: ', w05)
            w15 = w15 + dw15
            print ('weight 15: ', w15)
            w25 = w25 + dw25
            print ('weight 25: ', w25)
            w35 = w35 + dw35
            print ('weight 35: ', w35)

            # node a6 weight updates
            dw06 = 0.1 * error6 * 0
            dw16 = 0.1 * error6 * x1[iterationno]
            dw26 = 0.1 * error6 * x2[iterationno] 
            dw36 = 0.1 * error6 * x3[iterationno]
            w06 = w06 + dw06
            print ('weight 06: ', w06)
            w16 = w16 + dw16
            print ('weight 16: ', w16)
            w26 = w26 + dw26
            print ('weight 26: ', w26)
            w36 = w36 + dw36
            print ('weight 36: ', w36)

            # node a7 weight updates
            dw07 = 0.1 * error7 * 0
            dw47 = 0.1 * error7 * exponent4
            dw57 = 0.1 * error7 * exponent5
            dw67 = 0.1 * error7 * exponent6
            w07 = w07 + dw07
            print ('weight 07: ', w07)
            w47 = w47 + dw47
            print ('weight 47: ', w47)
            w57 = w57 + dw57
            print ('weight 57: ', w57)
            w67 = w67 + dw67
            print ('weight 67: ', w67)

            # node a8 weight updates
            dw08 = 0.1 * error8 * 0
            dw48 = 0.1 * error8 * exponent4
            dw58 = 0.1 * error8 * exponent5
            dw68 = 0.1 * error8 * exponent6
            w08 = w08 + dw08
            print ('weight 08: ', w08)
            w48 = w48 + dw48
            print ('weight 48: ', w48)
            w58 = w58 + dw58
            print ('weight 58: ', w58)
            w68 = w68 + dw68
            print ('weight 68: ', w68)
            
            epochs -= 1
            iterationno += 1
            testing -= 1
            if iterationno == 6:
                iterationno = 0
            if counter > 0:
                # calculating the softmax probabilities of both outputs 
                softmaxy1 = math.exp(0.7855907247339537) / (math.exp(0.7855907247339537) + math.exp(0.5173847648101272))
                softmaxy2 = math.exp(0.5173847648101272) / (math.exp(0.7855907247339537) + math.exp(0.5173847648101272))

                print ('softmax output y1: ', softmaxy1)
                print ('softmax output y2: ', softmaxy2)
                
        iterationno = 0
        
        # testing input data
        x1 = [0.3]
        x2 = [0.7]
        x3 = [0.9]
        
        epochs = 1
        counter += 1
        testing -= 1
        if testing > 0:
            print (' ')
            print (' ')
            print ('testing data: ')

training(100,[0.5,1,1,-0.01,0.5,0.01], [1,0.5,1,0.5,0.25,0.02], [0.75,0.75,1,0.25,0.13,0.05])
