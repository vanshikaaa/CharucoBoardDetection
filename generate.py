import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
from cv2 import aruco
from random import random, randint
import os
import pandas as pd

# charuco parameters
size = 0.2             # size of every box in meters
dimX = 6               # number of boxes per row
dimY = 6               # number of boxes per column
wideX = dimX * 200     # size of x board in pix
wideY = dimY * 200     # size of y board in pix
randomness = 500       # as this number increases, the charuco board gets more distorted
tol = 100              # increase this to make the board smaller
scale = 5              # how much the output images will be scaled down by
dict = aruco.Dictionary_get(aruco.DICT_4X4_250)  # change line 29 if this isn't 4x4

# other parameters
experimenterName = "vanshika"                 # deeplabcut experimenter
fname = "CollectedData_" + experimenterName   # names for csv and h5 files
sourceDir = "sample"                          # directory with background images
outputDir = "result"                          # directory for output images
projDir = "labeled-data/vid2"                 # deeplabcut path (labeled-data/videoname)

# assumes 4x4 charuco markings
charuco = aruco.CharucoBoard_create(dimX, dimY, size, 0.75*size, dict) 
img = charuco.draw((wideX, wideY))

empty = np.zeros((wideY, wideX,3))
empty[:, :, 0] = img/255
empty[:, :, 1] = img/255
empty[:, :, 2] = img/255

# array with coords of each inner intersection
originalPts = []
for y in range(1, dimY):
    thisRow = []
    for x in range(1, dimX):
        thisRow.append([(wideX/dimX) * x - 0.5, (wideY/dimY) * y - 0.5])
    originalPts.append(thisRow)
originalPts = np.array(originalPts)
print("charuco board created")

# max or min possible val of offset is randomness
def offset(): 
        return randomness * 2 * (random() - 0.5) 

# total number of sample files
total = len([f for f in os.listdir(sourceDir)if os.path.isfile(os.path.join(sourceDir, f))]) 

# names of output files
frames = ["image" + str(s) + ".png" for s in range(1, total + 1)]

# total number of "body parts"
totalPts = (dimX - 1) * (dimY - 1)

for index, bodypart in enumerate(list(map(str, range(1, totalPts + 1)))):
        columnindex = pd.MultiIndex.from_product(
            [[experimenterName], [bodypart], ["x", "y"]], names=["scorer", "bodyparts", "coords"]
        )
        frame = pd.DataFrame(
            100 + np.ones((len(frames), 2)) * 50 * index,
            columns=columnindex,
            index=[os.path.join(projDir, fn) for fn in frames],
        )
        if index == 0:
            dataFrame = frame
        else:
            dataFrame = pd.concat([dataFrame, frame], axis=1)


for count, filename in enumerate(os.listdir(sourceDir)):

    # rotate the charuco board a random amount of times
    for i in range (randint(0, 3)):
        empty = cv2.rotate(empty, cv2.ROTATE_90_CLOCKWISE)
        originalPts = np.rot90(originalPts)

    # tol keeps the charuco board at least tol pixels away from the edge of frame
    # offset() is the random change
    # randomness is the max value offset will be, so adding/subtracting it will make sure the point won't be out of the frame
    pt_A = [      -tol + offset() - randomness,       -tol + offset() - randomness]
    pt_B = [      -tol + offset() - randomness, wideY +tol + offset() + randomness]
    pt_C = [wideX +tol + offset() + randomness, wideY +tol + offset() + randomness]
    pt_D = [wideX +tol + offset() + randomness,       -tol + offset() - randomness]
    
    # math for width and height of transformed image
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1], [maxWidth - 1, 0]])

    # generate perspective matrix
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    # apply perspective matrix
    out = cv2.warpPerspective(empty,M,(maxWidth, maxHeight),flags=cv2.INTER_NEAREST, borderValue=2) 
    transPts = cv2.perspectiveTransform(originalPts, M)

    bkgd = mpimg.imread(sourceDir + "/" + filename)
    bkgd = cv2.resize(bkgd, (out.shape[1], out.shape[0])) / 255

    # border has value of 2, so mark and replace all border pixels with new bkgd
    mask = out[:, :, 0] == 2
    out[mask] = bkgd[mask]
    
    # convert out to the right format then save it as an image
    out = cv2.resize( out, (int(out.shape[1]/scale), int(out.shape[0]/scale)) )
    out = out * 255
    out = out.astype(np.uint8)
    mpimg.imsave(outputDir + "/image" + str(count + 1) + ".png", out)

    transPts = transPts.reshape(totalPts, 2) / scale
    for i, val in enumerate(transPts):
        dataFrame.loc[projDir + "/image" + str(count + 1) + ".png", (experimenterName, str(i + 1), "x")] = val[0]
        dataFrame.loc[projDir + "/image" + str(count + 1) + ".png", (experimenterName, str(i + 1), "y")] = val[1]


    print(" {}/{} images generated, {}%".format(count + 1, total, round(((count + 1)/total) * 100, 1)), end="\r")

print("{}/{} images generated, 100%                                      ".format(total, total))

print("saving data...")

dataFrame.to_hdf(fname + ".h5", "df_with_missing", format="table", mode="w",)
dataFrame.to_csv(fname + ".csv")

print("finished")

#                      fun color maps: tab20, Pastel1, spring, seismic, rainbow, magma, gnuplot
#    maps that actually work properly: gist_gray, cubehelix, nipy_spectral, gist_heat, hot, gnuplot2, gist_earth, CMRmap