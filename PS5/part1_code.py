from pymagnitude import *
file_path = "GoogleNews-vectors-negative300.magnitude"
vectors = Magnitude(file_path)

print("q1: ")
print(vectors.dim)

print("q2:")
print(vectors.most_similar(vectors.query("cat"), topn=6))

print("q3:")
print(vectors.doesnt_match(['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']))

print("q4:")
print(vectors.most_similar(positive = ["leg", "throw"], negative=["kick"]))