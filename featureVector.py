import numpy as np


class FeatureExtractor:

    @staticmethod
    def unique(list):
        unique_list=np.array(list)
        return np.unique(unique_list)

    @staticmethod
    def getUniqueFeatures(list):
        usl=np.sort(FeatureExtractor.unique(list))
        return usl

    
class FeatureVector:
    
    @staticmethod
    def getFeatureVector(usl,recipes):
        listFV=[]

        for recipe in recipes:
            fv=[]
            target,ingredients=recipe
            for feature in usl:
                if feature not in ingredients:
                    fv.append(0)
                else:
                    fv.append(1)
            listFV.append((target, fv))

        return listFV
