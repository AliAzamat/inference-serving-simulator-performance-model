# Predictions, written before the comparison

These numbers come from servesim/model.py alone. Nothing here was measured. The
simulator has not been run against them yet. Step 8 puts the two side by side.

What the model says:

    mean prompt 862 tokens, mean answer 105 tokens
    most requests per second the server can finish: 12.17
     rate  batch  step ms  latency s   tok/s
      1.0    1.3     12.2       1.37     105
      4.0    6.6     13.2       1.70     420
      8.0   19.8     15.6       2.52     839
     10.0   32.9     18.0       3.33    1049
     12.0   58.7     22.7       4.94    1259

Read it as a bet. Past about 12 requests per second the model says the server
cannot keep up. Below that it says latency grows slowly at first and then fast.
If the simulator disagrees, one of the two is wrong and the gap says which.
