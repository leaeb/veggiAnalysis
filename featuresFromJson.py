import json
art='fleisch/'
dat='Gefluegel.json'
features='gefluegelFeatures.txt'
f=open(art+dat)
data = json.load(f)
#print(data)

x=0
list=[]
for recipe in data:
    try:
        if(recipe["ingredients"]):
            for ingredient in recipe["ingredients"]:
                list.append(ingredient["name"])
    except KeyError:
        print(recipe)


cleared_list=[]
s=''
for item in list:
    #print(item)
    s=str(str(item).split(',')[0])
    s=str(s.split(' ')[0])
    s=str(s.split('(')[0])   
    s=str(s.split('z.B.')[0])
    s=str(s.split('oder')[0])
    s=str(s.split('für')[0])
    s=str(s.split('fuer')[0])
    s=str(s.split('evtl.')[0])
    s=str(s.split('ca.')[0])
    s=str(s.split(' g ')[0])
    s=str(s.split(' kg ')[0])
    s=str(s.split('EL')[0])
    s=s.strip()
    s=s.upper()
    cleared_list.append(s)

with open(art+features, 'w') as fp:
    for item in cleared_list:
        # write each item on a new line
        fp.write("%s,\n" % item)
    print('Done')





        
