# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:30:57 2017

@author: Administrator
"""

from PIL import Image
import struct
import os

def read_image(filename):
    f = open(filename,"rb")
    index = 0
    buf = f.read()
    f.close()
    
    magic,images,rows,cols = struct.unpack_from(">IIII",buf,index)
    index +=struct.calcsize(">IIII")
    print(index,images,rows,cols)
    
    for i in range(images):
        image = Image.new("L",(cols,rows))
        for x in range(rows):
            for y in range(cols):
                image.putpixel((y,x),int(struct.unpack_from(">B",buf,index)[0]))
                index+= struct.calcsize(">B")
        print("save"+str(i)+"image")
def read_label(filename, saveFilename):
  f = open(filename, 'rb')
  index = 0
  buf = f.read()
  f.close()
  magic, labels = struct.unpack_from('>II' , buf , index)
  index += struct.calcsize('>II')
  
  labelArr = [0] * labels
  #labelArr = [0] * 2000
  for x in range(labels):
  #for x in xrange(2000):
    labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
    index += struct.calcsize('>B')
  save = open(saveFilename, 'w')
  save.write(','.join(map(lambda x: str(x), labelArr)))
  save.write('\n')
  save.close()
  print("save labels success")       
#read_image("./data/train-images.idx3-ubyte")
#read_label("./data/train-labels.idx1-ubyte","./label.txt")
def read_save(style,modelname,labelname):
    f = open(modelname,"rb")
    modelindex = 0
    modelbuf = f.read()
    f.close()
    magic,images,rows,cols = struct.unpack_from(">IIII",modelbuf,modelindex)
    modelindex +=struct.calcsize(">IIII")
    print(modelindex,images,rows,cols)
    
    f = open(labelname, 'rb')
    labelindex = 0
    labelbuf = f.read()
    f.close()
    magic, labels = struct.unpack_from(">II" , labelbuf , labelindex)
    labelindex += struct.calcsize(">II")
    print(labels)
    
    if(style== "train"):
        prefix = "./mnist/train/"
    elif(style == "test"):
        prefix = "./mnist/test/"
    else:
        assert(0)
    assert(labels ==images)
    for l in range(labels):
        image = Image.new("L",(cols,rows))
        for x in range(rows):
            for y in range(cols):
                image.putpixel((y,x),int(struct.unpack_from(">B",modelbuf,modelindex)[0]))
                modelindex+= struct.calcsize(">B")
        labelvalue = int(struct.unpack_from('>B', labelbuf, labelindex)[0])
        labelindex += struct.calcsize('>B')
        print(labelvalue)
        image.save(prefix+str(labelvalue)+"/"+str(l)+".jpg")
if __name__ == '__main__':
    modelname = "./data/train-images.idx3-ubyte"
    labelname = "./data/train-labels.idx1-ubyte"
    for i in range(10):
        path = "./mnist/train/"+str(i)
        if(not os.path.exists(path)):
            os.makedirs(path)
        path = "./mnist/test/"+str(i)
        if(not os.path.exists(path)):
            os.makedirs(path)
#    read_save("train",modelname,labelname)
    modelname = "./data/t10k-images.idx3-ubyte"
    labelname = "./data/t10k-labels.idx1-ubyte"
    read_save("test",modelname,labelname)





        