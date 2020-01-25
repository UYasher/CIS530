#!/bin/bash

# Homework 1 Bash Question 2: Print Results from Subdirectories
# This is an individual homework.
# Implement the following function.
# You should make sure answer works from a Linux/Mac shell,
# and then paste them here for submission.

# Conditionally prints results from files in subdirectories.
#
#   Checks for a file named results.txt in each directory in $1.
#   For each existing results.txt, prints the directory name and then
#   only all lines from the corresponding results.txt that contain
#   the following substring:
#
#       Accuracy:
#
#   For example, the directory structure
#
#       root_directory/
#           1/
#               results.txt
#           2/
#           3/
#               results.txt
#
#   may output:
#       1
#       Accuracy: 54.44
#       Accuracy: 52.23
#       3
#       Accuracy: 44.34
#       Accuracy: 45.34
#
#   You should use ls, if statements, for statements, grep, and echo (and pipes.)

# Your solution will involve, in some way, listing the contents of the directory at the path $1.

for i in $( ls $1 ); do
    if [ -f $1/$i/results.txt ]; then
        echo "$i";
        grep "Accuracy:" $1/$i/results.txt;
    fi
done
