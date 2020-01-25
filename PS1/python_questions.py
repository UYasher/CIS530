'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

def check_for_foo_or_bar(text):
   '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
   '''

   bar = r"\b[bB][aA][rR]\b"
   foo = r"\b[fF][oO][oO]\b"

   if re.search(bar, text) is not None:
      if re.search(foo, text) is not None:
         return True
   return False


def replace_rgb(text):
   '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''

   is8bitNum = r" *(0*[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
   isDecNum = r" *((0\.[0-9]*)|(1\.0*)|(0|1))"
   isHexNum = r"([0-9]|[a-f]|[A-F])"

   isHexColor = r"(#((6HexDigits)|(3HexDigits)))"
   isHexColor = isHexColor.replace("3HexDigits", isHexNum*3)
   isHexColor = isHexColor.replace("6HexDigits", isHexNum*6)
   #print("isHexColor: " + isHexColor)

   isRGBcolor = r"rgb\(((8bitNum,8bitNum,8bitNum *)|(Dec,Dec,Dec *))\)"
   isRGBcolor = isRGBcolor.replace("8bitNum", is8bitNum)
   isRGBcolor = isRGBcolor.replace("Dec", isDecNum)
   #print("isRGBcolor: " + isRGBcolor)

   isColor = r"(?<!\S)(RGBcolor|HexColor)(?!\S)"
   isColor = isColor.replace("RGBcolor", isRGBcolor)
   isColor = isColor.replace("HexColor", isHexColor)
   #print("isColor: " + isColor)

   return re.sub(isColor, "COLOR", text)


def edit_distance(str1, str2):
  '''Computes the minimum edit distance between the two strings.

  Use a cost of 1 for all operations.

  See Section 2.4 in Jurafsky and Martin for algorithm details.
  Do NOT use recursion.

  Returns:
    An integer representing the string edit distance
    between str1 and str2
  '''

  n = len(str1)
  m = len(str2)
  d = [[0 for x in range(m+1)] for y in range(n+1)]

  for i in range(1, n+1):
    d[i][0] = d[i-1][0] + 1
  for j in range(1, m+1):
    d[0][j] = d[0][j-1] + 1

  for i in range(1, n+1):
    for j in range(1, m+1):
      d[i][j] = min(
        d[i-1][j] + 1,
        d[i-1][j-1] + (0 if str1[i-1] == str2[j-1] else 1),
        d[i][j-1] + 1
      )

  print(d)

  return d[n][m]


def wine_text_processing(wine_file_path, stopwords_file_path):
  '''Process the two files to answer the following questions and output results to stdout.

  1. What is the distribution over star ratings?
  2. What are the 10 most common words used across all of the reviews, and how many times
     is each used?
  3. How many times does the word 'a' appear?
  4. How many times does the word 'fruit' appear?
  5. How many times does the word 'mineral' appear?
  6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
     In natural language processing, we call these common words "stop words" and often
     remove them before we process text. stopwords.txt gives you a list of some very
     common words. Remove these stopwords from your reviews. Also, try converting all the
     words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
     different words). Now what are the 10 most common words across all of the reviews,
     and how many times is each used?
  7. You should continue to use the preprocessed reviews for the following questions
     (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
     reviews, and how many times is each used? 
  8. What are the 10 most used words among the 1 star reviews, and how many times is
     each used? 
  9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
     "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
     "white" reviews?
  10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

  No return value.
  '''

  # Init
  star_count = [0, 0, 0, 0, 0, 0]
  word_dict = {}
  processed_word_dict = {}
  five_star_dict = {}
  one_star_dict = {}
  red_dict = {}
  white_dict = {}
  stopwords = []
  num_reds = 0

  red = re.compile(r"(?<!\S)red(?!\S)")
  white = re.compile(r"(?<!\S)white(?!\S)")

  with open(stopwords_file_path, "r") as stopwords_file:
    for stopword in stopwords_file:
      stopwords.append(stopword.strip())

  # Process file
  with open(wine_file_path, "r", errors="ignore") as reviews:
    for line in reviews:
      review_text, stars = line.split("\t")
      stars = stars.strip()
      star_count[stars.count("*")-1] += 1

      # Naive counting
      for word in review_text.split(" "):
        increment(word_dict, word)

      # Intelligent counting
      review_text = review_text.lower()

      for word in review_text.split(" "):
        if word not in stopwords:
          increment(processed_word_dict, word)
          if len(stars) == 5:
            increment(five_star_dict, word)

          if len(stars) == 1:
            increment(one_star_dict, word)

          if red.search(review_text) is not None:
            increment(red_dict, word)

          if white.search(review_text) is not None:
            increment(white_dict, word)

  # Print Results

  # Question 1
  for i in range(6, 0, -1):
    print("*" * i + "\t"  + str(star_count[i - 1]))
  print()

  # Question 2
  ten_most_common(word_dict)

  # Question 3
  print(word_dict["a"])
  print()

  # Question 4
  print(word_dict["fruit"])
  print()

  # Question 5
  print(word_dict["mineral"])
  print()

  # Question 6
  ten_most_common(processed_word_dict)

  # Question 7
  ten_most_common(five_star_dict)

  # Question 8
  ten_most_common(one_star_dict)

  # Question 9
  ten_most_common(red_dict, not_in=white_dict)

  # Question 10
  ten_most_common(white_dict, not_in=red_dict)


# HELPER FUNCTIONS:
def ten_most_common(dictionary, not_in=None):
    # TODO: Need to find a way of sorting alphabetically -- maybe sort the collection using a custom function instead of using 10 most common
    if not_in:
        unique_keys = set(dictionary) - set(not_in)
        unique_dictionary = {key:dictionary[key] for key in unique_keys}
    else:
        unique_dictionary = dictionary
    sorted_keys = sorted(unique_dictionary, key=lambda x: (-unique_dictionary[x], x))
    for i in range(10):
        print(sorted_keys[i] + "\t" + str(unique_dictionary[sorted_keys[i]]))
    print()


def increment(dictionary, word):
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1