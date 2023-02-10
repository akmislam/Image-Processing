# ESE 558 - Digital Image Processing
# Project Part 1: Histogram Specification
# Akm Islam

# Libraries to be used.
import matplotlib.pyplot as plt
from PIL import Image


########### Read Images as HSV ###########
print("Begin reading images.")
fi = Image.open(r"Images/cookie.jpg").convert('HSV')     # Open image and convert it into HSV. Store it as 'fi'.
fo = Image.open(r"Images/cookie2.jpeg").convert('HSV')     # Open image and convert it into HSV. Store it as 'fo'.
M, N = fi.size                                          # Retrieve dimensions of f.
fih, fis, fiv = fi.split()                              # Split the "input" image into H, S, V components.
foh, fos, fov = fo.split()                              # Split the "output" image into H, S, V components.

print("Images are now read as HSV with intensity component.")


############ Obtain Histograms and A(i,0/1/2) Values ###########
A = [[0] * 5] * M * N                               # Initialize array A.
hi = [0] * 256                                      # Initialize histogram for input image.
ho = [0] * 256                                      # Initialize histogram for output image.
index = 0                                           # Initialize index for array A
for i in range (M):             
    for j in range (N):
        pixel_fi = fiv.getpixel((i,j))
        pixel_fo = fov.getpixel((i,j))
        hi[pixel_fi] += 1                           # Increment count for particular pixel for fi.
        ho[pixel_fo] += 1                           # Increment count for particular pixel for fo.
        A[index][0] = pixel_fi                      # Store pixel value in A[i][0].
        A[index][1] = i                             # Store m index of f(m,n) in A[i][1].
        A[index][2] = j                             # Store n index of f(m,n) in A[i][2].
        index = index + 1                           # Increment index to store next pixel characteristics.
print("Histograms are obtained with array A.")


########### Compute Partial Sums of Histograms ###########
ci = [0] * 256                                      # Initialize array for input histogram CDF.
co = [0] * 256                                      # Initialize array for output histogram CDF.
for i in range(256):
    if (i == 0):
        ci[i] = hi[i]                               # Initialize "input" ci to first element.
        co[i] = ho[i]                               # Initialize "output" co to first element.
    else:
        ci[i] = hi[i] + ci[i-1]                     # Add previous element with current histogram value.
        co[i] = ho[i] + ci[i-1]                     # Add previous element with current histogram value.
print("Partial Sums are computed.")


########### Sort Pixels into A(i,3) ###########
print("Sorting pixels into A(i,3)")
index = 0                                           # Reset index value.
j = 0                                               # Create counter for index of output histogram.
while(j < 256):                                     # For the entire output histogram.
    for k in range(ho[j]):                           
        A[index][3] = j                             # Assign the ho[j] j's into A[index][3].
        index = index + 1                           # Increment index.
    j += 1                                          # Increment j.
print("Pixels are sorted into array A(i,3).")


########### Equality Constraint ###########
for k in range (256):                               # For each possible grey-level.
    if (k == 0):                                    # Handle special case k == 0,
        imin = 0                                    
    else:
        imin = ci[k-1]                              
    sum = 0                                         # Initialize sum to 0.
    for i in range(imin, ci[k]):                    # For each pixel in the input image with intensity k which are in the order c[k-1] to c[k] - 1in the sorted list A[i,0] == k,
        sum += A[i][3]                              # find the average of the output gray-levels of A[i,3]. All those pixels will be set to the average value.
    if (hi[k] == 0):                                # Note, this is using A[i,3] not A[i,0].
        avg = 0                                     # Handle special case when hi[k] is 0.
    else:                                           # Otherwise,
        avg = round((sum/hi[k]))                    # Note hi[k] = c[k] = c[k-1]. Round-off fractions to nearest integer.
    for i in range(imin, ci[k]):                    # hi[k] gives the number of pixels added.
        A[i][4] = avg                               # Set all those pixels to the average value.
print("Equality constraint was employed.")


########### Reconstruct Desired Output Image ###########
gv = fov.copy()                                     # Create new image variable for desired image output.
for i in range (M*N):
    gv.putpixel((A[i][1],A[i][2]), A[i][4])         # Insert new pixel value into desired image output.
ghsv = Image.merge('HSV', (fih, fis, gv))           # Merge H, S, and new V channels into one image.
g = ghsv.convert("RGB")                             # Convert into RGB image.
g.save("output.jpg")                                # Save image as output.png.
print("Desired output image reconstructed.")


########### Plotting Images ###########
# Plot Input Image.
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Input Image (dull)")
plt.imshow(fi)
plt.show()

# Plot Input Image.
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Input Image (bright)")
plt.imshow(fo)
plt.show()

# Plot Output Image.
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Output Image")
plt.imshow(g)
plt.show()