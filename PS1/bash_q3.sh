#!/bin/bash

# Homework 1 Bash Question 3: Print Results from Subdirectories
# This is an individual homework.
# Implement the following function.
# You should make sure answer works from a Linux/Mac shell,
# and then paste them here for submission.

# Uses bash to print accuracies from a results file at $1.
#
#   Frequently, when dealing with large sets of experiments, you want
#   to summarize a bunch of semi-structured results text files quickly.
#   In this exercise, you'll use bash to take results of the form found
#   at
#        $1
#   and pull out the accuracies as well as the name of the experiment.
#
#   For example, the line
#
#       Base accuracy: 0.3267522959523133 time: .4555
#
#   should be transformed to the line
#
#       Base 0.3267522959523133
#
#   You should use cat, grep, and cut (and pipes.)

cat $1 | grep "accuracy:" | cut -d " " -f 1,3
