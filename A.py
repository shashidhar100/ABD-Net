f = open("cup_0005.txt", "r")

l = []

content = f.readlines()


for line in content:
      
    for i in line:
        print(i) 
        # Checking for the digit in 
        # the string
        # if i.isdigit() == True:
              
            # a += int(i)
f.close() 