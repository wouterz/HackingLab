from PIL import Image, ImageDraw
i = 3
j = 3
grid = [];
while(i<110):
    while(j<110):
        grid.append([(i,j),(i+35,j+35)])
        j=j+35
        j=j+4;
    i=i+4;
    j=3
    i=i+35
print(grid)

#for i in grid:
#    
#    img = Image.new('RGB', (120,120), color='black')
#    draw = ImageDraw.Draw(img)
#    draw.rectangle(i, fill='white')
#    img.save('pil' +str(i)+ '.png')

img = Image.new('RGB', (120,120), color='black')
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
for i in ls:
    img = Image.new('RGB', (120,120), color='black')
    size = len(i)
    for t in i:
        draw = ImageDraw.Draw(img)
        draw.rectangle(t, fill='white')
    squareCount[size]=squareCount[size]+1;
    img.save('./images/square/squares'+str(size)+'_'+str(squareCount[size])+'.png')
