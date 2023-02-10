# ESE 558 - Digital Image Processing
# Project Part 5: Spatial Domain Liner Filtering
# Akm Islam

########### Libraries ##########
import numpy as np
import math
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image


### 2D Cyldricial Filter Function
def Cylindrical2DFilter(x,y,radius):                    # Declare function. Pass parameters x, y, and radius.
    if (x**2 + y**2 <= radius):                         # If the point is inside the circle,
        return 1                                        # Output is 1.
    else:                                               # Otherwise,
        return 0                                        # Output is 0.


### Gaussian Filter Function
def GaussianFilter(x,y, sigma):                         # Declare function. Pass parameters x, y and sigma.                                                   
    return ((1/(2*math.pi*sigma**2))*(math.e**(-(x**2+y**2)/(2*sigma**2))))   # Gaussian Function (2D)


### Convolution with 2D Cylindrical Filter Function
def ConvolutionAlgorithmC(M,N,Radius,f,g):              # Declare function. Pass parameters M, N (image dimensions), Radius, image f(input), and image g (output).           
    for m in range (M):                                 # For all m in M.
        for n in range (N):                             # For all n in N.
            sum = 0                                     # Initialize sum to 0.
            weight = 0                                  # Initialize weight to 0.
            for p in range (-Radius, Radius+1):         # For all p in -Radius to Radius + 1.
                if (m-p < 0):                           # If p is out of bounds,
                    k = abs(m-p)                        # Reflect image.
                elif (m-p > M-1):                       # If p is out of bounds,
                    k = M-1-((m-p)-(M-1))               # Reflect image.
                else:                                   # Otherwise,
                    k = m-p                             # Obtain k as difference of m and p.
                for q in range (-Radius, Radius+1):     # For all m in -Radius to Radius + 1.
                    if (n-q < 0):                       # If q is out of bounds,
                        l = abs(n-q)                    # Reflect image.
                    elif (n-q > N-1):                   # If q is out of bounds,
                        l = N-1-((n-q)-(N-1))           # Reflect image.
                    else:                               # Otherwise,
                        l = n-q                         # Obtain l as difference of n and q.
                    c = Cylindrical2DFilter(p,q,Radius) # Calculate value of filter at point (p,q).
                    if (c):                             # If the point is within the circle,
                        weight += 1                     # Count how many points are summed.
                    sum += c*f[l][k]                    # Add product to sum.
            g[n][m] = round(sum/weight)                 # Find average value of sum.
    return g                                            # Return output image array.


### Convolution with Gaussian Filter Function
def ConvolutionAlgorithmG(M,N,Radius,f,g):              # Declare function. Pass parameters M, N (image dimensions), Radius, image f(input), and image g (output).
    for m in range (M):                                 # For all m in image.
        for n in range (N):                             # For all n in image.
            sum = 0                                     # Initialize sum to 0.
            for p in range (-Radius, Radius+1):         # For all p in filter.
                if (m-p < 0):                           # If point is out of bounds,
                    k = abs(m-p)                        # Assign new value.
                elif (m-p > M-1):                       # If point is out of bounds,
                    k = M-1-((m-p)-(M-1))               # Assign new value.
                else:                                   # Otherwise,
                    k = m-p                             # Obtain k as difference of m and p.
                for q in range (-Radius, Radius+1):     # For all q in filter.
                    if (n-q < 0):                       # If point is out of bounds,
                        l = abs(n-q)                    # Assign new value.
                    elif (n-q > N-1):                   # If point is out of bounds,
                        l = N-1-((n-q)-(N-1))           # Assign new value.
                    else:                               # Otherwise,
                        l = n-q                         # Obtain l as difference of n and q.
                    sum += round(GaussianFilter(p,q,Radius)*f[l][k])   # Calculate product.
            g[n][m] = sum                               # Insert summated value inside output array.
    return g                                            # Return output



########## Reading Image ##########
I1 = Image.open(r"Images/dance2gray.jpg").convert('L')  # Read image as grey-scale.
M,N = I1.size                                           # Obtain image dimensions.
f = asarray(I1)                                         # Convert Input image to numpy array.

### Plot Input Image
plt.title("Input Image")                                # Title plot.
plt.imshow(I1, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.



########## Initializing Output Images ##########
G1 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g1 = asarray(G1)                                        # Convert to numpy array.

G2 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g2 = asarray(G2)                                        # Convert to numpy array.

G3 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g3 = asarray(G3)                                        # Convert to numpy array.

G4 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g4 = asarray(G4)                                        # Convert to numpy array.

G5 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g5 = asarray(G5)                                        # Convert to numpy array.

G6 = Image.new('L',(M,N))                               # Create new Grey-scale image.
g6 = asarray(G6)                                        # Convert to numpy array.



########## Compute Convolution with each filter with different Radius values ##########
### 2D Cylindrical Convolutions
g1 = ConvolutionAlgorithmC(M,N,1,f,g1)                  # Filter image with 2D Cylindrical filter, radius 1.
g2 = ConvolutionAlgorithmC(M,N,3,f,g2)                  # Filter image with 2D Cylindrical filter, radius 3.
g3 = ConvolutionAlgorithmC(M,N,5,f,g3)                  # Filter image with 2D Cylindrical filter, radius 5.

### Gaussian Convolutions
g4 = ConvolutionAlgorithmG(M,N,1,f,g4)                  # Filter image with Gaussian filter, radius 1.
g5 = ConvolutionAlgorithmG(M,N,3,f,g5)                  # Filter image with Gaussian filter, radius 3.
g6 = ConvolutionAlgorithmG(M,N,5,f,g6)                  # Filter image with Gaussian filter, radius 5.



########## Assemble Images ##########
G1 = Image.fromarray(g1)                                # Reconstruct output 1.
G2 = Image.fromarray(g2)                                # Reconstruct output 2.
G3 = Image.fromarray(g3)                                # Reconstruct output 3.
G4 = Image.fromarray(g4)                                # Reconstruct output 4.
G5 = Image.fromarray(g5)                                # Reconstruct output 5.
G6 = Image.fromarray(g6)                                # Reconstruct output 6.

### Saving Images
G1.save("Images/2DCylindrical_Radius_1.jpg")            # Save image.
G2.save("Images/2DCylindrical_Radius_3.jpg")            # Save image.
G3.save("Images/2DCylindrical_Radius_5.jpg")            # Save image.
G4.save("Images/Gaussian_Radius_1.jpg")                 # Save image.
G5.save("Images/Gaussian_Radius_3.jpg")                 # Save image.
G6.save("Images/Gaussian_Radius_5.jpg")                 # Save image.

### Plot Output Images
plt.title("2D Cylindrical Filter, Radius = 1")          # Title plot.
plt.imshow(G1, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

plt.title("2D Cylindrical Filter, Radius = 3")          # Title plot.
plt.imshow(G2, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

plt.title("2D Cylindrical Fitler, Radius = 5")          # Title plot.
plt.imshow(G3, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

plt.title("Gaussian Filter, Radius = 1")                # Title plot.
plt.imshow(G4, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

plt.title("Gaussian Filter, Radius = 3")                # Title plot.
plt.imshow(G5, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.

plt.title("Gaussian Filter, Radius = 5")                # Title plot.
plt.imshow(G6, cmap='gray')                             # Pass image.
plt.show()                                              # Display image.