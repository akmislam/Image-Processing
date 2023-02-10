# ESE 558 - Digital Image Processing
# Project Part 2: Geometric Transformations
# Akm Islam

# Libraries to be used.
import math
from math import ceil, floor
from operator import invert
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray


### Function to implement Bilienar Interpolation. Equation from project document.
def bilinear(dx, dy, s1, s2, s3, s4):                   # Pass dx, dy, s1=Z1, s2=Z3, s3=Z2, s4=Z4.
    Z12 = dx*(int(s3)-int(s1))+s1                       # Interpolate Z12.
    Z34 = dx*(int(s2)-int(s4))+s2                       # Interpolate Z34.
    Z1234 = dy*(int(Z34)-int(Z12))+Z12                  # Interpolate Z1234.
    return Z1234                                        # Return value.

### Function to implement Gaussian Interpolation. 
def  GaussianFilter(x,y):                               # Pass x and y coordinates to the function
    sigma = 1.0                                         # Set value of sigma.
    return ((1/(2*math.pi*sigma**2))*(math.e**(-(x**2 + y**2) / (2 * sigma**2))))   # Gaussian Function (2D)



########### Read Images as RGB ###########
I1 = Image.open(r"Images/food1.jpg").convert('RGB')     # Open input image and convert to RGB channels.
M, N = I1.size                                          # Retrieve dimensions of image.

### Plot Input Image
plt.title("Input Image")                                # Title plot.
plt.imshow(I1)                                          # Pass image.
plt.show()                                              # Display image.

I1R, I1G, I1B = I1.split()
I1R_array = asarray(I1R)                                # Convert Red channel to numpy array.
I1G_array = asarray(I1G)                                # Convert Green channel to numpy array.
I1B_array = asarray(I1B)                                # Convert Blue channel to numpy array.




########## Configure Affine Matrix for Rotation ##########
theta = 57.0                                            # Set rotation angle (degrees).
thetar = theta * np.pi / 180.0                          # Convert angle to radians.
A = np.array([[np.cos(thetar), -np.sin(thetar)],        # Set Affine Matrix to rotation.
              [np.sin(thetar)      , np.cos(thetar) ]])
T = np.transpose(np.array([100,100]))                   # Set Translation.
Ap = np.linalg.inv(A)                                   # Calculate inverse of matrix.  

########## Determine New Coordinates ##########
p = np.dot(A, np.transpose(np.array([0,0]))) + T        # First corner point.
x1 = p[0]
y1 = p[1]
p = np.dot(A, np.transpose(np.array([0,N-1]))) + T      # Second corner point.
x2 = p[0]
y2 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,0]))) + T      # Third corner point.
x3 = p[0]
y3 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,N-1]))) + T    # Fourth corner point.
x4 = p[0]
y4 = p[1]

xmin = floor(min(x1,x2,x3,x4))                          # Determine minimum x value.
xmax = ceil(max(x1,x2,x3,x4))                           # Determine maximum x value.
ymin = floor(min(y1,y2,y3,y4))                          # Determine minimum y value.
ymax = ceil(max(y1,y2,y3,y4))                           # Determine maximum y value.
Mp = ceil(xmax-xmin) + 1                                # Determine M length of image.
Np = ceil(ymax-ymin) + 1                                # Determine N length of image.



########## Create New Images ##########
### Zoom with Bilinear Interpolation Image
I2 = Image.new('RGB',(Mp,Np))                           # Create new RGB image.
I2R, I2B, I2G = I2.split()                              # Split the channels of image.
I2R_array = asarray(I2R)                                # Convert Red channel to numpy array.
I2G_array = asarray(I2G)                                # Convert Green channel to numpy array.
I2B_array = asarray(I2B)                                # Convert Blue channel to numpy array.

### Zoom with Gaussian Interpolation Image
I3 = Image.new('RGB',(Mp,Np))                           # Create new RGB image.
I3R, I3B, I3G = I3.split()                              # Split the channels of image.
I3R_array = asarray(I3R)                                # Convert Red channel to numpy array.
I3G_array = asarray(I3G)                                # Convert Green channel to numpy array.
I3B_array = asarray(I3B)                                # Convert Blue channel to numpy array.



########## Filter Parameters ##########
k = 2                                                   # Parameter for filter size.
normalization_factor = 1.2                              # Normalization factor of filter.


########## Interpolation of Pixel Values ##########
def Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, BR, BG, BB, GR, GG, GB):
    for i in range(xmin,xmax+1):                            # Scan through new image to interpolate pixel values.
        for j in range(ymin, ymax+1):
            p = np.dot(Ap, np.transpose(np.array([i, j])) - T)  # Calculate which original coordinates to map to.
            x0 = p[0]                                       # Determine x-coordinate.
            y0 = p[1]                                       # Determine y-coordinate.
            xn = round(x0)                                  # Round x0.
            yn = round(y0)                                  # Round y0.
            xc = x0 - xn                                    # Determine x offset.
            yc = y0 - yn                                    # Determine y offset.

            if( (0 <= xn) and (xn <= M-1) and (0 <= yn) and (yn <= N-1)):   # Ensure that pixel value is within bounds.
                x = round(i-xmin)                           # Calculate x.
                y = round(j-ymin)                           # Calculate y.

                # Using Bilinear Interpolation
                minx = floor(x0)                            # Determine minimum x value.
                maxx = ceil(x0)                             # Determine maximum x value.
                miny = floor(y0)                            # Determine minimum y value.
                maxy = ceil(y0)                             # Determine maximum y value.
                dx = x0 - minx                              # Determine offset.
                dy = y0 - miny                              # Determine offset.
                if (maxx == M):                             # If maxx is out of bounds,
                    maxx = floor(x0)                        # Use floor instead.
                if (maxy == N):                             # If maxy is out of bounds,
                    maxy = floor(y0)                        # Use floor instead.
                # Red
                s1 = I1R_array[miny][minx]                  # Obtain Z1 red.
                s2 = I1R_array[maxy][minx]                  # Obtain Z3 red.
                s3 = I1R_array[miny][maxx]                  # Obtain Z2 red.
                s4 = I1R_array[maxy][maxx]                  # Obtain Z4 red.
                BR[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.
                
                # Green
                s1 = I1G_array[miny][minx]                  # Obtain Z1 green.
                s2 = I1G_array[maxy][minx]                  # Obtain Z3 green.
                s3 = I1G_array[miny][maxx]                  # Obtain Z2 green.
                s4 = I1G_array[maxy][maxx]                  # Obtain Z4 green.
                BG[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.

                # Blue
                s1 = I1B_array[miny][minx]                  # Obtain Z1 blue.
                s2 = I1B_array[maxy][minx]                  # Obtain Z3 blue.
                s3 = I1B_array[miny][maxx]                  # Obtain Z2 blue.
                s4 = I1B_array[maxy][maxx]                  # Obtain Z4 blue.
                BB[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.

                # Using the Convolution Interpolation Filter (Gaussian)
                sumR, sumG, sumB = 0,0,0                    # Initialize sums (for each channel).
                for m1 in range(-k,k+1):                    # Obtain values across filter.
                    for n1 in range(-k,k+1):
                        gx = xn-m1                          # Calculate x-coordinate.
                        gy = yn-n1                          # Calculate y-coordinate.
                        if (gx < 0):                        # If x-coordinate is too small,
                            gx = 0                          # Set to 0.
                        if (gx > M-1):                      # If x-coordinate is too large,
                            gx = M-1                        # Set to M-1.
                        if (gy < 0):                        # If y-coordinate is too small,
                            gy = 0                          # Set to 0.
                        if (gy > N-1):                      # If y-coordinate is too large,
                            gy = N-1                        # Set to N-1.
                        sampleValueR = I1R_array[gy][gx]    # Obtain Red value.
                        sampleValueG = I1G_array[gy][gx]    # Obtain Green value.
                        sampleValueB = I1B_array[gy][gx]    # Obtain Blue value.
                        filterCoeff = (1.0/normalization_factor) * GaussianFilter(m1-xc,n1-yc)  # Calculate filter coefficient.
                        sumR += filterCoeff*sampleValueR    # Sum Red value.
                        sumG += filterCoeff*sampleValueG    # Sum Green value.
                        sumB += filterCoeff*sampleValueB    # Sum Blue value.
                GR[y][x] = sumR                             # Insert red pixel value.
                GG[y][x] = sumG                             # Insert green pixel value.
                GB[y][x] = sumB                             # Insert blue pixel value.
    return BR, BG, BB, GR, GG, GB



### Run for Zoomed out Images
I2R_array, I2G_array, I2B_array, I3R_array, I3G_array, I3B_array = Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, I2R_array, I2G_array, I2B_array, I3R_array, I3G_array, I3B_array)



########## Configure Affine Matrix for Zoom Out ###########
# Zoom
zoom = 1.2
A = np.array([[zoom, 0],                                # Set Affine Matrix to zoom.
              [0, zoom]])
T = np.transpose(np.array([0,0]))                       # Set Translation.

Ap = np.linalg.inv(A)                                   # Calculate inverse of matrix.  

########## Determine New Coordinates ##########
p = np.dot(A, np.transpose(np.array([0,0]))) + T        # First corner point.
x1 = p[0]
y1 = p[1]
p = np.dot(A, np.transpose(np.array([0,N-1]))) + T      # Second corner point.
x2 = p[0]
y2 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,0]))) + T      # Third corner point.
x3 = p[0]
y3 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,N-1]))) + T    # Fourth corner point.
x4 = p[0]
y4 = p[1]

xmin = floor(min(x1,x2,x3,x4))                          # Determine minimum x value.
xmax = ceil(max(x1,x2,x3,x4))                           # Determine maximum x value.
ymin = floor(min(y1,y2,y3,y4))                          # Determine minimum y value.
ymax = ceil(max(y1,y2,y3,y4))                           # Determine maximum y value.
Mp = ceil(xmax-xmin) + 1                                # Determine M length of image.
Np = ceil(ymax-ymin) + 1                                # Determine N length of image.



########## Create New Images ##########
### Rotation with Bilinear Interpolation Image
I4 = Image.new('RGB',(Mp,Np))                           # Create new RGB image.
I4R, I4B, I4G = I4.split()                              # Split the channels of image.
I4R_array = asarray(I4R)                                # Convert Red channel to numpy array.
I4G_array = asarray(I4G)                                # Convert Green channel to numpy array.
I4B_array = asarray(I4B)                                # Convert Blue channel to numpy array.

I4d = Image.new('RGB',(Mp,Np))                          # Create new RGB image.
I4dR, I4dB, I4dG = I4d.split()                          # Split the channels of image.
I4dR_array = asarray(I4dR)                              # Convert Red channel to numpy array.
I4dG_array = asarray(I4dG)                              # Convert Green channel to numpy array.
I4dB_array = asarray(I4dB)                              # Convert Blue channel to numpy array.

### Run for Rotated Images
I4R_array, I4G_array, I4B_array, a, b, c = Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, I4R_array, I4G_array, I4B_array, I4dR_array, I4dG_array, I4dB_array)

########## Configure Affine Matrix for Zoom In ##########
zoom = 0.5
A = np.array([[zoom, 0],                                # Set Affine Matrix to zoom.
              [0, zoom]])
T = np.transpose(np.array([0,0]))                       # Set Translation.
Ap = np.linalg.inv(A)                                   # Calculate inverse of matrix.                         


########## Determine New Coordinates ##########
p = np.dot(A, np.transpose(np.array([0,0]))) + T       # First corner point.
x1 = p[0]
y1 = p[1]
p = np.dot(A, np.transpose(np.array([0,N-1]))) + T     # Second corner point.
x2 = p[0]
y2 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,0]))) + T     # Third corner point.
x3 = p[0]
y3 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,N-1]))) + T   # Fourth corner point.
x4 = p[0]
y4 = p[1]

xmin = floor(min(x1,x2,x3,x4))                         # Determine minimum x value.
xmax = ceil(max(x1,x2,x3,x4))                          # Determine maximum x value.
ymin = floor(min(y1,y2,y3,y4))                         # Determine minimum y value.
ymax = ceil(max(y1,y2,y3,y4))                          # Determine maximum y value.
Mp = ceil(xmax-xmin) + 1                               # Determine M length of image.
Np = ceil(ymax-ymin) + 1                               # Determine N length of image.

### Rotation with Gaussian Interpolation Image
I5 = Image.new('RGB',(Mp,Np))                          # Create new RGB image.
I5R, I5B, I5G = I5.split()                             # Split the channels of image.
I5R_array = asarray(I5R)                               # Convert Red channel to numpy array.
I5G_array = asarray(I5G)                               # Convert Green channel to numpy array.
I5B_array = asarray(I5B)                               # Convert Blue channel to numpy array.

I5d = Image.new('RGB',(Mp,Np))                         # Create new RGB image.
I5dR, I5dB, I5dG = I5d.split()                         # Split the channels of image.
I5dR_array = asarray(I5dR)                             # Convert Red channel to numpy array.
I5dG_array = asarray(I5dG)                             # Convert Green channel to numpy array.
I5dB_array = asarray(I5dB)                             # Convert Blue channel to numpy array.

I5R_array, I5G_array, I5B_array, a, b, c = Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, I5R_array, I5G_array, I5B_array, I5dR_array, I5dG_array, I5dB_array)


########## Configure Affine Matrix for Rotation ##########
theta = 57.0                                            # Set rotation angle (degrees).
thetar = theta * np.pi / 180.0                          # Convert angle to radians.
A = np.array([[np.cos(thetar), -np.sin(thetar)],        # Set Affine Matrix to rotation.
              [np.sin(thetar)      , np.cos(thetar) ]])
T = np.transpose(np.array([100,100]))                   # Set Translation.
Ap = np.linalg.inv(A)                                   # Calculate inverse of matrix.  


########## Determine New Coordinates ##########
p = np.dot(A, np.transpose(np.array([0,0]) - np.transpose(np.array([int(M/2), int(N/2)])))) + T   + np.transpose(np.array([int(M/2), int(N/2)]))      # First corner point.
x1 = p[0]
y1 = p[1]
p = np.dot(A, np.transpose(np.array([0,N-1]) - np.transpose(np.array([int(M/2), int(N/2)])))) + T     + np.transpose(np.array([int(M/2), int(N/2)]))  # Second corner point.
x2 = p[0]
y2 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,0]) - np.transpose(np.array([int(M/2), int(N/2)])))) + T   + np.transpose(np.array([int(M/2), int(N/2)]))    # Third corner point.
x3 = p[0]
y3 = p[1]
p = np.dot(A, np.transpose(np.array([M-1,N-1]) - np.transpose(np.array([int(M/2), int(N/2)])))) + T  + np.transpose(np.array([int(M/2), int(N/2)]))   # Fourth corner point.
x4 = p[0]
y4 = p[1]

xmin = floor(min(x1,x2,x3,x4))                          # Determine minimum x value.
xmax = ceil(max(x1,x2,x3,x4))                           # Determine maximum x value.
ymin = floor(min(y1,y2,y3,y4))                          # Determine minimum y value.
ymax = ceil(max(y1,y2,y3,y4))                           # Determine maximum y value.
Mp = ceil(xmax-xmin) + 1                                # Determine M length of image.
Np = ceil(ymax-ymin) + 1                                # Determine N length of image.


########## Create New Images ##########
### Zoom in with Bilinear Interpolation Image
I6 = Image.new('RGB',(Mp,Np))                           # Create new RGB image.
I6R, I6B, I6G = I6.split()                              # Split the channels of image.
I6R_array = asarray(I6R)                                # Convert Red channel to numpy array.
I6G_array = asarray(I6G)                                # Convert Green channel to numpy array.
I6B_array = asarray(I6B)                                # Convert Blue channel to numpy array.

I6d = Image.new('RGB',(Mp,Np))                          # Create new RGB image.
I6dR, I6dB, I6dG = I6d.split()                          # Split the channels of image.
I6dR_array = asarray(I6dR)                              # Convert Red channel to numpy array.
I6dG_array = asarray(I6dG)                              # Convert Green channel to numpy array.
I6dB_array = asarray(I6dB)                              # Convert Blue channel to numpy array.


def Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, BR, BG, BB):
    for i in range(xmin,xmax+1):                            # Scan through new image to interpolate pixel values.
        for j in range(ymin, ymax+1):
            p = np.dot(Ap, np.transpose(np.array([i- int(M/2), j-int(N/2)])) - T) + np.transpose(np.array([int(M/2), int(N/2)]))  # Calculate which original coordinates to map to.
            x0 = p[0]                                       # Determine x-coordinate.
            y0 = p[1]                                       # Determine y-coordinate.
            xn = round(x0)                                  # Round x0.
            yn = round(y0)                                  # Round y0.
            xc = x0 - xn                                    # Determine x offset.
            yc = y0 - yn                                    # Determine y offset.

            if( (0 <= xn) and (xn <= M-1) and (0 <= yn) and (yn <= N-1)):   # Ensure that pixel value is within bounds.
                x = round(i-xmin)                           # Calculate x.
                y = round(j-ymin)                           # Calculate y.

                # Using Bilinear Interpolation
                minx = floor(x0)                            # Determine minimum x value.
                maxx = ceil(x0)                             # Determine maximum x value.
                miny = floor(y0)                            # Determine minimum y value.
                maxy = ceil(y0)                             # Determine maximum y value.
                dx = x0 - minx                              # Determine offset.
                dy = y0 - miny                              # Determine offset.
                if (maxx == M):                             # If maxx is out of bounds,
                    maxx = floor(x0)                        # Use floor instead.
                if (maxy == N):                             # If maxy is out of bounds,
                    maxy = floor(y0)                        # Use floor instead.
                # Red
                s1 = I1R_array[miny][minx]                  # Obtain Z1 red.
                s2 = I1R_array[maxy][minx]                  # Obtain Z3 red.
                s3 = I1R_array[miny][maxx]                  # Obtain Z2 red.
                s4 = I1R_array[maxy][maxx]                  # Obtain Z4 red.
                BR[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.
                
                # Green
                s1 = I1G_array[miny][minx]                  # Obtain Z1 green.
                s2 = I1G_array[maxy][minx]                  # Obtain Z3 green.
                s3 = I1G_array[miny][maxx]                  # Obtain Z2 green.
                s4 = I1G_array[maxy][maxx]                  # Obtain Z4 green.
                BG[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.

                # Blue
                s1 = I1B_array[miny][minx]                  # Obtain Z1 blue.
                s2 = I1B_array[maxy][minx]                  # Obtain Z3 blue.
                s3 = I1B_array[miny][maxx]                  # Obtain Z2 blue.
                s4 = I1B_array[maxy][maxx]                  # Obtain Z4 blue.
                BB[y][x] = bilinear(dx,dy,s1,s2,s3,s4)      # Calculate bilinear interpolation, insert Red pixel value.
    return BR, BG, BB



### Run for Zoomed In Images
I6R_array, I6G_array, I6B_array = Interpolate(Ap, T, xmin, xmax, ymin, ymax, Mp, Np, I6R_array, I6G_array, I6B_array)



########## Assemble Output Images ##########
I2R = Image.fromarray(I2R_array)                        # Convert numpy array to image object (Red)  
I2G = Image.fromarray(I2G_array)                        # Convert numpy array to image object (Green)
I2B = Image.fromarray(I2B_array)                        # Convert numpy array to image object (Blue)       
I2 = Image.merge('RGB', (I2R, I2G, I2B))                # Combine RGB channels to single image object.                   
I2.save("Images/Rotation_Translation_Bilinear.jpg")     # Save image.

I3R = Image.fromarray(I3R_array)                        # Convert numpy array to image object (Red)  
I3G = Image.fromarray(I3G_array)                        # Convert numpy array to image object (Green)
I3B = Image.fromarray(I3B_array)                        # Convert numpy array to image object (Blue)       
I3 = Image.merge('RGB', (I3R, I3G, I3B))                # Combine RGB channels to single image object.                   
I3.save("Images/Rotation_Translation_Gaussian.jpg")     # Save image.

I4R = Image.fromarray(I4R_array)                        # Convert numpy array to image object (Red)  
I4G = Image.fromarray(I4G_array)                        # Convert numpy array to image object (Green)
I4B = Image.fromarray(I4B_array)                        # Convert numpy array to image object (Blue)       
I4 = Image.merge('RGB', (I4R, I4G, I4B))                # Combine RGB channels to single image object.                   
I4.save("Images/ZoomOutBilinear.jpg")                   # Save image.

I5R = Image.fromarray(I5R_array)                        # Convert numpy array to image object (Red)  
I5G = Image.fromarray(I5G_array)                        # Convert numpy array to image object (Green)
I5B = Image.fromarray(I5B_array)                        # Convert numpy array to image object (Blue)       
I5 = Image.merge('RGB', (I5R, I5G, I5B))                # Combine RGB channels to single image object.                   
I5.save("Images/ZoomInBilinear.jpg")                    # Save image.

I6R = Image.fromarray(I6R_array)                        # Convert numpy array to image object (Red)  
I6G = Image.fromarray(I6G_array)                        # Convert numpy array to image object (Green)
I6B = Image.fromarray(I6B_array)                        # Convert numpy array to image object (Blue)       
I6 = Image.merge('RGB', (I6R, I6G, I6B))                # Combine RGB channels to single image object.                   
I6.save("Images/Rotation_New_Center.jpg")               # Save image.


########## Plot Images ##########
plt.title("Rotate with Bilinear Interpolation")         # Title plot.
plt.imshow(I2)                                          # Input image.
plt.show()                                              # Display image.

plt.title("Rotate with Gaussian Interpolation")         # Title plot.
plt.imshow(I3)                                          # Input image.
plt.show()                                              # Display image.

plt.title("Zoom Out with Bilinear Interpolation")       # Title plot.
plt.imshow(I4)                                          # Input image.
plt.show()                                              # Display image.

plt.title("Zoom In with Bilinear Interpolation")        # Title plot.
plt.imshow(I5)                                          # Input image.
plt.show()                                              # Display image.

plt.title("Rotation (Centered) Bilinear Interpolation") # Title plot.
plt.imshow(I6)                                          # Input image.
plt.show()                                              # Display image.