import re

file1 = open("Hamlet.txt", "r")
hamlet= file1.read()
words = re.split(",| |\.|\n|\?|\]|\[|\!|\-|\'|\"|:|;", hamlet)
file1.close()
file2 = open("HamletWords.txt", "w")
for w in words:
    if len(w) > 1:
        file2.write(str(w))
        file2.write("\n")

file2.close()