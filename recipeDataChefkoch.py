
from re import X
from chefkoch import ChefKochAPI,DataParser


#get all available dish categories
categories = ChefKochAPI.get_categories()
x=0
for category in categories:
    #print(category.title)
    if(category.title=='Geflügel'):
        y=x
    x=x+1
recipes = ChefKochAPI.parse_recipes(categories[y])

#write recipes to json file one at a time
DataParser.write_recipes_to_json(category.title, recipes)