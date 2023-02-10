# ESE 558 - Digital Image Processing
# Project Part 3: Spatial Domain Non-Liner Filtering
# Akm Islam

# Libraries to be used.
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image



### Median Filter Function
def MedianFilter(a, size):                            # Pass array and length to function.
    a.sort()                                          # Sort the array.
    return a[int (size/2)]                            # Return middle (median) value.



### KNN Filter Function                         
def KNNfilter(a, size, Kn):                           # Pass array, length, and number of neighbors to function.
    mean = 0                                          # Initialize the mean.
    center = a[int(size/2)]                           # Obtain value of center pixel.
    distance = [abs(int(center)-int(x)) for x in a]   # Subtract value from all pixels.
    i = 0                                             # Initialize number of neighbors found.
    while (i<Kn):                                     # While 4 values are not found, keep searching.
        for j in range(size):                         # Search through array.
            if (j == int(size/2)):                    # If index is center of filter,
                continue                              # Skip the iteration.
            if (distance[j] == 0):                    # If value is 0 (neighbor),
                mean = mean + a[j]                    # Add to the mean.
                i += 1                                # Increment neighbors found counter.
            if (i == Kn):                             # If Kn nearest neighbors are found,
                break                                 # Break out of for loop.
        distance = [abs(x-1) for x in distance]       # After searching, find next nearest neighbors and try again.
    return int(mean/Kn)                               # Return average of KNN values.



########## Parameters for Median Filter ##########
P = 2                                                 # Width of filter.
Q = 2                                                 # Length of filter.
totalM = (2*P+1)*(2*Q+1)                              # Number of elements.
Med = [0]*totalM                                      # Initialize the filter.



########## Parameters for KNN Filter ##########
R = 1                                                 # Width of filter.
S = 1                                                 # Length of filter.
totalK = (2*R+1)*(2*S+1)                              # Number of elements.
Kn = 4                                                # Number of neighbors.
Knn = [0]*totalK                                      # Initialize the filter.



########## Read input Image ##########
f = Image.open(r"Images/PCB.jpg").convert('HSV')     # Read input image as HSV.
M, N = f.size                                        # Obtain image dimensions.                                       
fH, fS, fV = f.split()                               # Split the HSV channels.
gm = fV.copy()                                       # Create new image variable to store Median Filter output.
gk = fV.copy()                                       # Create new image variable to store KNN Filter output.
fV_array = asarray(fV)                               # Make array version of input image Value component.
gm_array = fV_array.copy()                           # Make array version of Median Filter Value component.
gk_array = fV_array.copy()                           # Make array version of KNN Filter Value component.
k, l = 0,0                                           # Initialize k and l to 0 (global scope).



########## Median Filter ##########
for m in range (M):                                  # Scan through image.
    for n in range(N):
        r = 0                                        # Initialize Median Filter index to 0.
        for p in range(-P, P+1):                     # Scan through filter.
            if (m-p < 0):                            # If point is too far to the left,
                k = abs(m-p)                         # Reflect image.
            elif (m-p > M-1):                        # If point is too far to the right,
                k = M-1-((m-p)-(M-1))                # Reflect image.
            else:                                    # Otherwise,
                k = m-p                              # Obtain k coordinate.
            for q in range (-Q, Q+1):                # Scan through filter.
                if(n-q < 0):                         # If point is too far to the top,
                    l = abs(n-q)                     # Reflect image.
                elif(n-q > N-1):                     # If point is too far to the bottom,
                    l = N-1-((n-q)-(N-1))            # Reflect image.
                else:                                # Otherwise,
                    l = n-q                          # Obtain l coordinate.
                Med[r] = fV_array[k][l]              # Store the value inside the array.
                r += 1                               # Increment the index.
        gm_array[m][n] = MedianFilter(Med, totalM)   # Calculate output of Median Filter, store in output array.



########## KNN Filter ##########
for m in range (M):                                  # Scan through image.
    for n in range(N):
        d = 0                                        # Initialize KNN Filter index to 0.
        for r in range(-R, R+1):                     # Scan through filter.
            if (m-r < 0):                            # If point is too far to the left,
                k = abs(m-r)                         # Reflect image.
            elif (m-r > M-1):                        # If point is too far to the right,
                k = M-1-((m-r)-(M-1))                # Reflect image.
            else:                                    # Otherwise,
                k = m-r                              # Obtain k coordinate.
            for s in range (-S, S+1):                # Scan through filter.
                if(n-s < 0):                         # If point is too far to the top,
                    l = abs(n-s)                     # Reflect image.
                elif(n-s > N-1):                     # If point is too far to the bottom,
                    l = N-1-((n-s)-(N-1))            # Reflect image.
                else:                                # Otherwise,
                    l = n-s                          # Obtain l coordinate.
                Knn[d] = fV_array[k][l]              # Store the value inside the array.
                d += 1                               # Increment the index.
        gk_array[m][n] = KNNfilter(Knn, totalK, Kn)  # Calculate output of KNN Filter, store in output array.



########## Assembling the Median Filter Output ##########
gmV = Image.fromarray(gm_array)                      # Convert array to image object.
gMHSV = Image.merge('HSV', (fH, fS, gmV))            # Combine HSV components.
gM = gMHSV.convert("RGB")                            # Convert to RGB image.
gM.save("Images/Median_Filter_Output.jpg")           # Save image.



########## Assembling the KNN Filter Output ##########
gkV = Image.fromarray(gk_array)                      # Convert array to image object.
gKHSV = Image.merge('HSV', (fH, fS, gkV))            # Combine HSV components.
gK = gKHSV.convert("RGB")                            # Convert to RGB image.
gK.save("Images/KNN_Filter_Output.jpg")              # Save image.



########## Plotting Images ##########
plt.title("Input")                                   # Title plot.
plt.imshow(f)                                        # Input desired image.
plt.show()                                           # Show plot.


plt.title("Median Filter Output")                    # Title plot.
plt.imshow(gM)                                       # Input desired image.
plt.show()                                           # Show plot.


plt.title("KNN Filter Output")                       # Title plot.
plt.imshow(gK)                                       # Input desired image.
plt.show()                                           # Show plot.