import json
fleisch = 'fleisch/'
datGefl = 'Gefluegel.json'
fG = open(fleisch+datGefl)
dataGefl = json.load(fG)

vegan = 'vegan/'
datVegan = 'Vegan.json'
fV = open(vegan+datVegan)
dataVegan = json.load(fV)
# print(data)

x = 0
allRecipes = []


def featuresPerRecipe(isVegan, data):
    allRecipes = []
    for recipe in data:
        try:
            if(recipe["ingredients"]):
                list = []
                for ingredient in recipe["ingredients"]:
                    list.append(ingredient["name"])
        except KeyError:
            print(recipe)

        cleared_list = []
        s = ''
        for item in list:
            # print(item)
            s = str(str(item).split(',')[0])
            s = str(s.split(' ')[0])
            s = str(s.split('(')[0])
            s = str(s.split('z.B.')[0])
            s = str(s.split('oder')[0])
            s = str(s.split('für')[0])
            s = str(s.split('fuer')[0])
            s = str(s.split('evtl.')[0])
            s = str(s.split('ca.')[0])
            s = str(s.split(' g ')[0])
            s = str(s.split(' kg ')[0])
            s = str(s.split('EL')[0])
            s = s.strip()
            s = s.upper()
            cleared_list.append(s)
        allRecipes.append((isVegan, cleared_list))
    rec = []
    rec = allRecipes
    return rec


allRecipesG = featuresPerRecipe(False, dataGefl)
allRecipesV = featuresPerRecipe(True, dataVegan)

#print(allRecipesG[0])
#print(allRecipesV[1])

isVegan, incredients = allRecipesV[0];
print(isVegan)
print(incredients)
from main import usl
from featureVector import FeatureVector


fvV=FeatureVector.getFeatureVector(usl=usl,recipes=allRecipesV)

fvG=FeatureVector.getFeatureVector(usl=usl,recipes=allRecipesG)

print(len(allRecipesG))
print(len(allRecipesV))
fvs=fvV+fvG

import csv

with open('data.csv','w') as out:
    csv_out=csv.writer(out)
    for row in fvs:
        csv_out.writerow(row)

print(len(fvV))
print(len(fvG))
target,ing=fvV[3]
print(target)
print(len(ing))

