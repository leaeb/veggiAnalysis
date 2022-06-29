from featureVector import FeatureExtractor,FeatureVector
import numpy as np

featuresVegan = open("vegan/veganFeatures.txt", "r")
linesVegan = featuresVegan.read().split(',\n')

featuresGefl = open("fleisch/gefluegelFeatures.txt", "r")
linesGefl = featuresGefl.read().split(',\n')
print(len(linesVegan))
print(len(linesGefl))
print(len(linesVegan+linesGefl))

sl=FeatureExtractor.getUniqueFeatures(linesVegan + linesGefl)
print(len(sl))
usl=list(filter(None, sl)) 

print(usl)
