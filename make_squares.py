from PIL import Image, ImageDraw
import os
initI = 3
initJ = 3
squareSize = 5
i = initI
j = initJ
total = initI*5+(squareSize+1)*4
grid = []
while(i<total):
    while(j<total):
        grid.append([(i,j),(i+squareSize,j+squareSize)])
        j+=squareSize
        j+=initJ+1
    i+=initI+1
    j=initJ
    i+=squareSize
print(grid)

#for i in grid:
#    
#    img = Image.new('RGB', (120,120), color='black')
#    draw = ImageDraw.Draw(img)
#    draw.rectangle(i, fill='white')
#    img.save('pil' +str(i)+ '.png')

img = Image.new('RGB', (total,total), color='black')
for i in grid:
    draw = ImageDraw.Draw(img)
    draw.rectangle(i, fill='white')
ls = []
def powerset(s):
     x = len(s)
     for i in range(1 << x):
         ls.append([s[j] for j in range(x) if (i & (1 << j))])
powerset(grid)
j = 0
squareCount = [0,0,0,0,0,0,0,0,0,0]

try:
    os.stat("./images/square_35p")
except:
    os.makedirs("./images/square_35p")
for i in ls:
    img = Image.new('RGB', (total,total), color='black')
    size = len(i)
    if size<10:
        for t in i:
            draw = ImageDraw.Draw(img)
            draw.rectangle(t, fill='white')
        squareCount[size]=squareCount[size]+1
        img.save('./images/square_35p/squares_35p_'+str(size)+'_'+str(squareCount[size])+'.png')
