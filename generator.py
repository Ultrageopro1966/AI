from random import choice

with open('dataset.txt', 'w') as file:
    for _ in range(10000):
        wrt = ''
        array = [str(choice([0, 1])) for _ in range(3)]
        wrt+=''.join(array)
        if array[1] == '1' or array[2] == '1':
            wrt+=' 1\n'
        else:
            wrt+=' 0\n'
        file.write(wrt)