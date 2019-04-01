# Simple-neural-network-in-C++
A really simple toy code neural network in C++ (derived from Andrew Trask)

Andrew Trask's 'Neural Network in 11 lines of Python' was the thing that made machine learning approachable for me. I learn so much better having something to play around with.

This is my attempt to recreate it in C++ using the Eigen library. Eigen's used by the likes of Tensorflow, and functions a lot like Numpy in Python. Which makes this toy code almost as simple and readable.

See Trask's original Python implementation here (with full annotations and explanations):
http://iamtrask.github.io/2015/07/27/python-network-part2/

Download Eigen from here:
http://eigen.tuxfamily.org/index.php?title=Main_Page

If you're just building and running your C++ files from something simple like Sublime Text, all you need to do is put the Eigen subfolder (with things like Dense) in your working directory and change the header to commas:

#include "Eigen/Dense"

Then Cmd+B
