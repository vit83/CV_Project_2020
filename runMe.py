import os
import Estimator.BusFinder
import math
import time
def run(myAnnFileName, buses):
    ImageDirPath = buses
    ImageFiles = GetFilesFromDir(ImageDirPath, 'JPG')
    BoxDict = {}
    for Image in ImageFiles:
        print("analyzing: {:s}".format(Image))
        start = time.time()
        Boxes = Estimator.BusFinder.get_prediction(Image)
        end = time.time()
        elapsed = float(end) - float(start)
        s = "analysis time is %0.3f seconds" % elapsed
        print(s)
        if (Boxes is None):
            print("can not find objects in " + Image)
        elif Boxes.size != 0:
            BoxDict[Image] = Boxes
        else:
            #to do may add adaptive threshold and rerun estimation
            print("can not find objects in " + Image)

    WriteResults(BoxDict,myAnnFileName)



def GetFilesFromDir(DirPath, filetpye):
    Files = []
    for r, d, f in os.walk(DirPath):
        for file in f:
            if filetpye in file.upper():
                Files.append(os.path.join(r, file))
    return Files


def WriteResults(Results,file):
    with open(file, 'w+') as result_file:
        for  imageFullPath, Boxlist  in Results.items():
            FileName = os.path.basename(imageFullPath)
            line = FileName + ":"
            for box in Boxlist:
                boxString = "[{:d},{:d},{:d},{:d},{:d}],".format(math.ceil(box[0]),math.ceil(box[1]),math.ceil(box[2]),math.ceil(box[3]),math.ceil(box[4]))
                line += boxString
            #remove last comma
            line = line[:-1] + '\n'
            result_file.writelines(line)





    pass

if __name__ == "__main__":
    run('myAnnFile.txt', 'busesTrain')

