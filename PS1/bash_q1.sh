#!/bin/bash

# Homework 1 Bash Question 1: Create a Vocab with Bash
# This is an individual homework.
# Implement the following function.
# You should make sure answer works from a Linux/Mac shell,
# and then paste them here for submission.

# Runs a shell command to output a vocab for the given text file, found at $1.
#
#    Assume unique words are separated either by a space, or are on separate
#    lines.
#
#    The output file should have two tab-separated columns, containing the
#    words sorted from least frequent to most frequent in the first
#    column, and the frequency of each word in the second column.
#
#    For example, for input:
#     Seven lazy researchers like using bash.
#     The researchers like, like Python too.
#
#    The output should be
#
#      bash.	1
#      lazy	1
#      like,	1
#      Python	1
#      Seven	1
#      The	1
#      too.	1
#      using	1
#      like	2
#      researchers	2
#
#    You should use sed, tr, sort, uniq, and awk (and pipes.)


# cat $1 prints the file at the path specified by $1,
# the 1-indexed command line argument.
# Your solution should start with this.

cat $1 | tr [:space:]  "\n" | sort | uniq -c | sort -n | awk '{print $2 "\t" $1}'

