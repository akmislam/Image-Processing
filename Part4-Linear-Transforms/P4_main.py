# ESE 558 - Digital Image Processing
# Project Part 4: Linear Transforms
# Akm Islam

# Libraries to be used.
import numpy as np

########## Input M and N Dimensions ##########
M = int(eval(input("Enter M:")))                            # Prompt user to enter M.
N = int(eval(input("Enter N:")))                            # Prompt user to enter N.

f = np.zeros((M,N))                                         # Initialize f to MxN matrix with 0's.
h = np.zeros((M,N,M,N))                                     # Initialize h to MxNxMxN matrix with 0's.

ga = np.zeros((M,N))                                        # Initialize ga (output for part A) to MxN matrix with 0's.
gb = np.zeros((M,N))                                        # Initialize gb (output for part B) to MxN matrix with 0's.
gc = np.zeros((M,N))                                        # Initialize gc (output for part C) to MxN matrix with 0's.
gd = np.zeros((M,N))                                        # Initialize gd (output for part D) to MxN matrix with 0's.

for m in range (M):                                         # For each m.
    for n in range (N):                                     # For each n.
        f[m][n] = np.random.randint(0,100+1)                # Randomly insert value into f.
        for u in range (M):                                 # For each u.
            for v in range (N):                             # For each v.
                h[m][n][u][v] = np.random.randint(0,100+1)  # Randomly insert value into h.



########## Part A ##########
for u in range (M):                                         # For each u.
    for v in range (N):                                     # For each v.
        for m in range (M):                                 # For each m.
            for n in range (N):                             # For each n.
                ga[u][v] += h[m][n][u][v]*f[m][n]           # Insert summated value into ga.



########## Part B ##########
h1 = np.zeros((M,M))                                        # Initialize h1 to MxM matrix with 0's.
h2 = np.zeros((N,N))                                        # Initialize h2 to NxN matrix with 0's.
g1 = np.zeros((M,N))                                        # Initialize g1m to MxN matrix with 0's.

for u in range (M):                                         # For each u.
    for m in range (M):                                     # For each m.
        h1[u][m] = np.random.randint(0,100+1)               # Randomly insert value into h1.

for v in range (N):                                         # For each v.
    for n in range (N):                                     # For each n.
        h2[v][n] = np.random.randint(0,100+1)               # Randomly insert value into h2.

for u in range (M):                                         # For each u.
    for v in range(N):                                      # For each v.
        for m in range (M):                                 # For each m.
            for n in range (N):                             # For each n.
                g1[m][v] += h2[v][n]*f[m][n]                # g1m accumulates summation.
            gb[u][v] += h1[u][m]*g1[m][v]                   # Insert summated value into gb.                      



########## Part C ##########
hc = np.zeros((M,N))                                        # Initialize hc to MxN matrix with 0's.
for m in range (M):                                         # For each m.
    for n in range (N):                                     # For each n.
        hc[m][n] = np.random.randint(0,100+1)               # Randomly insert value into hc.

for u in range (M):                                         # For each u.
    for v in range (N):                                     # For each v.
        for m in range (M):                                 # For each m.
            for n in range (N):                             # For each n.
                gc[u][v] += hc[(u-m+M)%M][(v-n+N)%N]*f[m][n]# Insert circular convolution value.



########## Part D ##########
h1c = np.zeros((M))                                         # Initialize h1c to M vector with 0's.
h2c = np.zeros((N))                                         # Initialize h2c to N vector with 0's.
g1 = np.zeros((M,N))                                        # Initialize g1 to MxN matrix with 0's.

for u in range (M):                                         # For each u.
    for v in range (N):                                     # For each v.
        for m in range (M):                                 # For each m.
            for n in range (N):                             # For each n.
                g1[m][v] += h2c[(v-n+N)%N]*f[m][n]          # Calculate seperable circular convolution.
            gd[u][v] += h1c[(u-m+M)%M]*g1[m][v]             # Insert circular convolution value.