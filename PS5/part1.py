from pymagnitude import *
file_path = "vectors/GoogleNews-vectors-negative300.magnitude"
vectors = Magnitude(file_path)

print("q1: ")
print(vectors.dim)

print("q2:")
print(vectors.most_similar(vectors.query("picnic"), topn=6)[1:])

print("q3:")
print(vectors.doesnt_match(['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette']))

print("q4:")
print(vectors.most_similar(positive = ["throw", "leg"], negative=["jump"])[0])
