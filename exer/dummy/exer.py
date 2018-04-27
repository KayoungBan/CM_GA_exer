# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:15:24 2018

@author: Bany
"""

# this is a comment
import math # imports make code from other modules available
# code blocks are initiated by a trailing colon followed by indented lines
class Circle: # define a class
  def __init__(self, radius): # constructor with parameter radius
  self.radius = radius # store the parameter in a class variable
  def get_area(self): # define a function that belongs to the class
  return math.pi \
  * self.radius \
  * self.radius # trailing \ continues the expression on the next line
# code that is not in a class is executed immediately
for i in range(1, 10):
  if (i & 1) == 0:
  continue
  circle = Circle(i) # create an instance
  print("A circle with radius {0} has area {1:0.2f}".format(
  i, circle.get_area() # `print` writes output to the console
  ))
