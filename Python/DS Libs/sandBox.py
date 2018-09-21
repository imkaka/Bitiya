
# # coding: utf-8

# # In[2]:


# def quicksort(arr):
#     if len(arr)<=1:
#         return arr
#     pivot= arr[len(arr)//2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x> pivot]
#     return quicksort(left) + middle + quicksort(right)

# print(quicksort([3,5,6,8,1,2,5,6,50]))


# # In[4]:


# s = "hello"
# print(s.capitalize())  # Capitalize a string; prints "Hello"
# print(s.upper())       # Convert a string to uppercase; prints "HELLO"
# print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
# print(s.center(7))     # Center a string, padding with spaces; prints " hello "
# print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
#                                 # prints "he(ell)(ell)o"
# print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"


# # In[5]:


# nums = list(range(5))     # range is a built-in function that creates a list of integers
# print(nums)               # Prints "[0, 1, 2, 3, 4]"
# print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
# print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
# print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
# print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
# print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
# nums[2:4] = [8, 9]        # Assign a new sublist to a slice
# print(nums)


# # In[6]:


# d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
# print(d['cat'])       # Get an entry from a dictionary; prints "cute"
# print('cat' in d)     # Check if a dictionary has a given key; prints "True"
# d['fish'] = 'wet'     # Set an entry in a dictionary
# print(d['fish'])      # Prints "wet"
# # print(d['monkey'])  # KeyError: 'monkey' not a key of d
# print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
# print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
# del d['fish']         # Remove an element from a dictionary
# print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# # In[8]:


# class Greeter(object):

#     # Constructor
#     def __init__(self, name):
#         self.name = name  # Create an instance variable

#     # Instance method
#     def greet(self, loud=False):
#         if loud:
#             print('HELLO, %s!' % self.name.upper())
#         else:
#             print('Hello, %s' % self.name)

# g = Greeter('Fred')  # Construct an instance of the Greeter class
# g.greet()            # Call an instance method; prints "Hello, Fred"
# g.greet(True)   # Call an instance method; prints "HELLO, FRED!"


# # In[11]:


# import numpy as np

# a = np.array([1, 2, 3])   # Create a rank 1 array
# print(type(a))            # Prints "<class 'numpy.ndarray'>"
# print(a.shape)            # Prints "(3,)"
# print(a[0], a[1], a[2])   # Prints "1 2 3"
# a[0] = 5                  # Change an element of the array
# print(a)                  # Prints "[5, 2, 3]"

# b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
# print(b.shape)                     # Prints "(2, 3)"
# print(b[0, 0], b[0, 1], b[1, 0])

# # Prints "1 2 4"

# l = [1,2,3]
# l


# # In[12]:


# # Create the following rank 2 array with shape (3, 4)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]]
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# # Use slicing to pull out the subarray consisting of the first 2 rows
# # and columns 1 and 2; b is the following array of shape (2, 2):
# # [[2 3]
# #  [6 7]]
# b = a[:2, 1:3]

# # A slice of an array is a view into the same data, so modifying it
# # will modify the original array.
# print(a[0, 1])   # Prints "2"
# b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"


# # In[15]:


# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# # Use slicing to pull out the subarray consisting of the first 2 rows
# # and columns 1 and 2; b is the following array of shape (2, 2):
# # [[2 3]
# #  [6 7]]
# b = a[:2, 1:3]

# # A slice of an array is a view into the same data, so modifying it
# # will modify the original array.
# print(a[0, 1])   # Prints "2"
# b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
# print(a[0, 1])   # Prints "77"
# #he original array. Note that this is quite different from the way that MATLAB handles array slicing:

# import numpy as np

# # Create the following rank 2 array with shape (3, 4)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]]
# a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# # Two ways of accessing the data in the middle row of the array.
# # Mixing integer indexing with slices yields an array of lower rank,
# # while using only slices yields an array of the same rank as the
# # original array:
# row_r1 = a[1, :]    # Rank 1 view of the second row of a
# row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
# print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
# print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# # We can make the same distinction when accessing columns of an array:
# col_r1 = a[:, 1]
# col_r2 = a[:, 1:2]
# print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
# print(col_r2, col_r2.shape)  # Prints "[[ 2]
#                              #          [ 6]
#                              #          [10]] (3, 1)"


# # In[18]:


# a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

# print(a)

# b= np.array([0,2,0,1])

# print(a[np.arange(4),b])


# # In[19]:


# a[np.arange(4), b] += 10

# print(a)


# # In[20]:


# print(a[a >6])


# # In[24]:


# x = np.array([[1,2],[3,4]])
# y = np.array([[5,6],[7,8]])
# print(x.shape)
# v = np.array([9,10])
# w = np.array([11, 12])
# print(v.shape)

# # Inner product of vectors; both produce 219
# print(v.dot(w))
# print(np.dot(v, w))

# # Matrix / vector product; both produce the rank 1 array [29 67]
# print(x.dot(v))
# print(np.dot(x, v))

# # Matrix / matrix product; both produce the rank 2 array
# # [[19 22]
# #  [43 50]]
# print(x.dot(y))
# print(np.dot(x, y))


# # In[25]:


# import numpy as np

# x = np.array([[1,2], [3,4]])
# print(x)    # Prints "[[1 2]
#             #          [3 4]]"
# print(x.T)  # Prints "[[1 3]
#             #          [2 4]]"

# # Note that taking the transpose of a rank 1 array does nothing:
# v = np.array([1,2,3])
# print(v)    # Prints "[1 2 3]"
# print(v.T)  # Prints "[1 2 3]"


# # In[26]:


# # Compute outer product of vectors
# v = np.array([1,2,3])  # v has shape (3,)
# w = np.array([4,5])    # w has shape (2,)
# # To compute an outer product, we first reshape v to be a column
# # vector of shape (3, 1); we can then broadcast it against w to yield
# # an output of shape (3, 2), which is the outer product of v and w:
# # [[ 4  5]
# #  [ 8 10]
# #  [12 15]]
# print(np.reshape(v, (3, 1)) * w)

# # Add a vector to each row of a matrix
# x = np.array([[1,2,3], [4,5,6]])
# # x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# # giving the following matrix:
# # [[2 4 6]
# #  [5 7 9]]
# print(x + v)

# # Add a vector to each column of a matrix
# # x has shape (2, 3) and w has shape (2,).
# # If we transpose x then it has shape (3, 2) and can be broadcast
# # against w to yield a result of shape (3, 2); transposing this result
# # yields the final result of shape (2, 3) which is the matrix x with
# # the vector w added to each column. Gives the following matrix:
# # [[ 5  6  7]
# #  [ 9 10 11]]
# print((x.T + w).T)
# # Another solution is to reshape w to be a column vector of shape (2, 1);
# # we can then broadcast it directly against x to produce the same
# # output.
# print(x + np.reshape(w, (2, 1)))

# # Multiply a matrix by a constant:
# # x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# # these can be broadcast together to shape (2, 3), producing the
# # following array:
# # [[ 2  4  6]
# #  [ 8 10 12]]
# print(x * 2)


# # In[42]:


# from scipy.misc import imread, imsave, imresize

# # Read an JPEG image into a numpy array
# img = imread('kaka.jpg')
# print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# # We can tint the image by scaling each of the color channels
# # by a different scalar constant. The image has shape (400, 248, 3);
# # we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# # numpy broadcasting means that this leaves the red channel unchanged,
# # and multiplies the green and blue channels by 0.95 and 0.9
# # respectively.
# img_tinted = img * [1, 0.95, 0.9]

# # Resize the tinted image to be 300 by 300 pixels.
# img_tinted = imresize(img_tinted, (300, 300))

# # Write the tinted image back to disk
# imsave('kaka_tinted.jpg', img_tinted)


# # In[34]:


# import numpy as np
# import matplotlib.pyplot as plt

# # Compute the x and y coordinates for points on a sine curve
# x = np.arange(0, 20, 0.5)
# print(x)
# y = np.sin(x)

# # Plot the points using matplotlib
# plt.plot(x, y)
# plt.show()


# # In[35]:


import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# # In[36]:


# import numpy as np
# import matplotlib.pyplot as plt

# # Compute the x and y coordinates for points on sine and cosine curves
# x = np.arange(0, 3 * np.pi, 0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)

# # Set up a subplot grid that has height 2 and width 1,
# # and set the first such subplot as active.
# plt.subplot(2, 1, 1)

# # Make the first plot
# plt.plot(x, y_sin)
# plt.title('Sine')

# # Set the second subplot as active, and make the second plot.
# plt.subplot(2, 1, 2)
# plt.plot(x, y_cos)
# plt.title('Cosine')

# # Show the figure.
# plt.show()


# # In[41]:


# import numpy as np
# from scipy.misc import imread, imresize
# import matplotlib.pyplot as plt

# img = imread('kaka.jpg')
# img_tinted = img * [1, 0.8, 0.9]

# # Show the original image
# plt.subplot(1, 2, 1)
# plt.imshow(img)

# # Show the tinted image
# plt.subplot(1, 2, 2)

# # A slight gotcha with imshow is that it might give strange results
# # if presented with data that is not uint8. To work around this, we
# # explicitly cast the image to uint8 before displaying it.
# plt.imshow(np.uint8(img_tinted))
# plt.show()
