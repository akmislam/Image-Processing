# ESE 558 - Digital Image Processing
# Project Part 6: Frequency Domain Filtering
# Akm Islam

########### Libraries ##########
import numpy as np
from numpy import asarray
import math
import cmath
import matplotlib.pyplot as plt
from PIL import Image

########## Initializations ##########
fi = Image.open(r"Images/food128.jpg").convert('L')     # Read image as grey-scale.
M,N = fi.size                                           # Obtain image dimensions.

plt.title("Input Image")                                # Title plot.
plt.imshow(fi, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

f = asarray(fi)                                         # Convert Input image to numpy array.
F = np.zeros((M,N), dtype = complex)                    # Initialize F to MxN matrix with 0's.

h = np.zeros((M,N), dtype = complex)                    # Initialize h to MxN matrix with 0's.
H = np.zeros((M,N), dtype = complex)                    # Initialize H to MxN matrix with 0's.

g = np.zeros((M,N), dtype = complex)                    # Initialize g to MxN matrix with 0's.
G = np.zeros((M,N), dtype = complex)                    # Initialize G to MxN matrix with 0's.

P = np.zeros((M,M), dtype = complex)                    # Initialize P to MxM matrix with 0's.
Pi = np.zeros((M,M), dtype = complex)                   # Initialize Pi to MxM matrix with 0's.
Q = np.zeros((N,N), dtype = complex)                    # Initialize Q to NxN matrix with 0's.
Qi = np.zeros((N,N), dtype = complex)                   # Initialize Qi to NxN matrix with 0's.


########## Insert FT and IFT Weights ###########
for u in range (M):                                     # For all u in M.
    for m in range (M):                                 # For all m in M.
        P[u][m] = (1/M)*cmath.exp(-math.pi*(u*m/M)*2j)  # Insert FT value.
        Pi[u][m] = cmath.exp(math.pi*(u*m/M)*2j)        # Insert IFT value.

for v in range (N):                                     # For all v in N.
    for n in range (N):                                 # For all n in N.
        Q[v][n] = (1/N)*cmath.exp(-math.pi*(v*n/N)*2j)  # Insert FT value.
        Qi[v][n] = cmath.exp(math.pi*(v*n/N)*2j)        # Insert IFT value.


########## Compute FT of F ##########
F = np.dot(np.dot(P,f),Q)                               # Compute Fourier Transform using matrix multiplication.


########## Assigning H Values ##########
for u in range (M):                                     # For all u in M.
    for v in range (N):                                 # For all v in N.
        if (((u<5) and (v<5)) or ((u>M-5) and (v<5)) or ((u<5) and (v>N-5)) or ((u>M-5) and (v>N-5))):  # If u or v is low frequency,
            H[u][v] = 0.5                               # Assign H[u][v] to 0.5.
        else:                                           # Otherwise,
            H[u][v] = 1                                 # Assign H[u][v] to 1.


########## Compute Output G ##########
G = F*H                                                 # Frequency Domain multiplication.

########## Compute Output g  ##########
g = np.dot(np.dot(Pi,G),Qi)                             # Compute Inverse Fourier Transform using matrix multiplication.
g = abs(g)                                              # Take absolute value of g.
g = g.astype(np.uint8)                                  # Type cast values to uint8.

### Assembling Images
gi = Image.fromarray(g)                                 # Reconstruct output image.
gi.save("Images/Output.jpg")                            # Save output image.

### Plot Output Images
plt.title("Output")                                     # Title plot.
plt.imshow(gi, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.