import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from PIL import ImageFilter
from PIL import Image as image

def GetImagesJpg(path):

   #Lists
   Group=[]
   filename_group=[]

   #All png files
   images = sorted (glob.glob(path+'/*.png'), key=len)

   for filename in images:
       im=cv2.imread(filename,0)

       #Images
       Group.append(im)

       #Filename
       filename_group.append(filename[len(path)+1:][:-4])

   return Group, filename_group

def Histogram(Image, Name):

    #Delete pixel = 0 to appreciate the histogram distribution better
    histo = Image[Image != 0]

    plt.figure()

    #Original Image
    plt.subplot(211), plt.imshow(Image,'gray'), plt.title('Original Image ' + Name),

    #Histogram
    plt.subplot(212), plt.hist(histo[histo != 255].ravel(), 256), plt.title('Histogram' + Name)

def conservative_smoothing_gray(data, filter_size):
    temp = []

    indexer = filter_size // 2

    new_image = data.copy()

    nrow, ncol = data.shape

    for i in range(nrow):

        for j in range(ncol):

            for k in range(i-indexer, i+indexer+1):

                for m in range(j-indexer, j+indexer+1):

                    if (k > -1) and (k < nrow):

                        if (m > -1) and (m < ncol):

                            temp.append(data[k,m])

            temp.remove(data[i,j])


            max_value = max(temp)

            min_value = min(temp)

            if data[i,j] > max_value:

                new_image[i,j] = max_value

            elif data[i,j] < min_value:

                new_image[i,j] = min_value

            temp =[]

    return new_image.copy()

def Unsharp(Image):
    Filter = image.fromarray(Image.astype('uint8'))
    Filter = Filter.filter(ImageFilter.UnsharpMask(radius=4, percent=250))
    return (Filter)

def Filtering(Image, Name):

    #Delete pixel = 0 to appreciate the histogram distribution better
    ksize = 3
    plt.figure()
    Filter = Image
    Filters = {0:'Original Image ', 1:'Mean Filter ', 2:'Gaussian Filter ', 3:'Median Filter ', 4:'Conservative Filter '}
    for i in range(5):
        plt.subplot(5,2,2*i+1), plt.imshow(Filter,'gray'), plt.title(Filters[i] + Name)
        plt.subplot(5,2,2*i+2), plt.imshow(Unsharp(Filter),'gray'), plt.title('Unsharp ' + Filters[i] + Name)

        if i == 0:
            Filter = cv2.blur(Image, (ksize, ksize))
        elif i == 1:
            Filter = cv2.GaussianBlur(Image, (ksize, ksize), 0)
        elif i == 2:
            Filter = cv2.medianBlur(Image, ksize)
        elif i == 3:
            Filter = conservative_smoothing_gray(Image, 4)

def Processed_Images(Light, Surf, Deep, Comp, Name):

    plt.figure()

    #Highlighted Image
    plt.subplot(221), plt.imshow(Light,'gray'), plt.title('Filtered Image'),

    #Ischemia Cores Image
    plt.subplot(222), plt.imshow(Comp,'gray'), plt.title('Possible Ischemia Cores'),

    #Surface Image
    plt.subplot(223), plt.imshow(Surf,'gray'), plt.title('Surface Information'),

    #Deep Image
    plt.subplot(224), plt.imshow(Deep,'gray'), plt.title('Deep Information'),

    #Title
    plt.suptitle(Name, fontsize=16)

    plt.show()

def Ischemia_Cores(Group, Name):
        #Lists needed
        Isch_Cores = []
        Deep_Images = []
        Surf_Images = []
        Comp_Counter = []

        #Kernels for the morphological operators
        kernel1 = np.ones((4,4), np.uint8)
        kernel2 = np.ones((3,3), np.uint8)
        deep_dif = 25

        for i, Image in enumerate(Group):

            Filtered_Image = cv2.GaussianBlur(np.array(Unsharp(Image)),(3,3),0)
            #Find the adaptative threshold
            otsu, _ = cv2.threshold(Filtered_Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            print (otsu)
            if otsu >= 125:
                deep_dif = 40
            elif otsu <= 100:
                deep_dif = 15

            if i == 0:
                deep_dif = 80

            #Surface and deep division
            _, thresh_surf = cv2.threshold(Filtered_Image, otsu, 255, cv2.THRESH_BINARY)
            _, thresh_mix = cv2.threshold(Filtered_Image, otsu, 255, cv2.THRESH_TOZERO_INV)
            _, thresh_deep = cv2.threshold(thresh_mix, otsu - deep_dif, 255, cv2.THRESH_BINARY)
            if i == 0:
                deep_dif = 25
            #Invert black and white so it can find the components of the images
            Inv_Thresh = np.invert(thresh_surf)

            #Morphological operators
            Morph_1 = cv2.morphologyEx(Inv_Thresh, cv2.MORPH_ERODE, kernel1)
            Morph = cv2.morphologyEx(Morph_1, cv2.MORPH_DILATE, kernel2)

            #Find possible ischemia cores
            _,Components = cv2.connectedComponents(Morph)

            #Plot the different processed images
            Processed_Images(Filtered_Image, thresh_surf, thresh_deep, Components, Name[i])

            #Lists
            Isch_Cores.append(Components)
            Deep_Images.append(thresh_deep)
            Surf_Images.append(thresh_surf)
            Comp_Counter.append(np.amax(Components))


        return Isch_Cores, Deep_Images, Surf_Images, Comp_Counter

def Core_coloring(Core_IMG, Deep_IMG, Comp_Number, Big_DMG, Ischemia_Range = [25, 50]):

    #Transform the image to RGB
    Core0 = np.array(Core_IMG, dtype=np.uint8)
    RGB_Core = cv2.cvtColor(Core0,cv2.COLOR_GRAY2RGB)
    RGB_Core_Color = RGB_Core
    #print(len(Core_IMG.ravel()))
    #List to determine the ischemia
    Comp_Mean = []

    #Color Area
    Red = 0
    #Orange = 0
    Yellow = 0
    Green = 0

    #Coloring loop
    for j in range(Comp_Number-1):

        Repetition_Count = 0
        Value_Count = 0

        for y in range(len(Core_IMG)):

            for x in range(len(Core_IMG[y])):
                if Core_IMG[y][x] == j+1:
                    Repetition_Count += 1
                    Value_Count += Deep_IMG[y][x]

        #Ischemia core value
        Ischemia_mean = (Value_Count/(Repetition_Count*255))
        Ischemia_percentage = Ischemia_mean*100

        #Ischemia core coloring:
        #print(Repetition_Count, int(len(Core_IMG.ravel())/100))
        if Repetition_Count < int(len(Core_IMG.ravel())/100):
            RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[101, 227, 76],RGB_Core_Color)

        elif Repetition_Count < int(len(Core_IMG.ravel())/50):
            RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[249, 226, 58],RGB_Core_Color)

        elif Big_DMG and Repetition_Count > int(len(Core_IMG.ravel())/3):
            RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[242, 51, 51],RGB_Core_Color)


        else:
        #Red
            if Ischemia_percentage <= Ischemia_Range[0]:
                RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[242, 51, 51],RGB_Core_Color)

        #Yellow
            elif Ischemia_Range[0] <= Ischemia_percentage <= Ischemia_Range[1]:
                RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[249, 226, 58],RGB_Core_Color)

        #Green
            else:
                RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[101, 227, 76],RGB_Core_Color)

        #Orange
#        elif Ischemia_Range[0] <= Ischemia_percentage <= Ischemia_Range[1]:
#
#            RGB_Core_Color = np.where(RGB_Core == [j+1, j+1, j+1],[253, 151, 54],RGB_Core_Color)


    #Color Area
    Red = np.sqrt(np.count_nonzero(RGB_Core_Color == 242))
    Yellow = np.sqrt(np.count_nonzero(RGB_Core_Color == 249))
    Green = np.sqrt(np.count_nonzero(RGB_Core_Color == 101))
    #Orange = np.sqrt(np.count_nonzero(RGB_Core_Color == 253))


    return RGB_Core_Color, Red, Yellow, Green

def Mean_coloring(Core_IMG, Deep_IMG, Comp_Number):

    #Transform the image to RGB
    Core0 = np.array(Core_IMG, dtype=np.uint8)
    RGB_Core_Color = Core0

    #List to determine the ischemia
    Comp_Mean = []

    #Coloring loop
    for j in range(Comp_Number-1):

        Repetition_Count = 0
        Value_Count = 0

        for y in range(len(Core_IMG)):

            for x in range(len(Core_IMG[y])):
                if Core_IMG[y][x] == j+1:
                    Repetition_Count += 1
                    Value_Count += Deep_IMG[y][x]

        #Ischemia core value
        Ischemia_mean = (Value_Count/(Repetition_Count*255))
        Ischemia_percentage = Ischemia_mean*100
        RGB_Core_Color = np.where(Core0 == (j+1), Ischemia_percentage, RGB_Core_Color)

    return RGB_Core_Color

def Vessel_Area(Group):

    #List
    Count = []

    #Count pixel = 0 after highlighted an inverted image so it only counts the vessels
    for Image in Group:
        N = np.count_nonzero(Image == 0, axis = 1)
        N_total = np.sum(N)
        Count.append(N_total)

    #Calculate the area
    Area = np.sqrt(np.array(Count))
    Max_Value = Area[0]
    Area_Norm = Area/Max_Value

    return Area_Norm

def Regression_Mean(Area):

    Regression_Line = []

    for i in range(13):
        Counter = 0
        Area_Sum = 0
        for j in range(len (Area)):
            if Area[j][i] != 0:
                Area_Sum += Area[j][i]
                Counter += 1
        Regression_Value = Area_Sum / Counter
        Regression_Line.append(Regression_Value)

    return Regression_Line

def Color_Cuantification (Group, Cores, Deeps, Comp, Ischemia_Range):

    Red_Density=[]
    Yellow_Density=[]
    Green_Density=[]
    RGB_Images=[]
    Shapes=[]


    for i in range(len(Group)):
        Big_DMG = False

        if 1 <= i <= 5 :
            Big_DMG = True

        RGB_Core_Color, Red, Yellow, Green = Core_coloring(Cores[i], Deeps[i], Comp[i], Big_DMG, Ischemia_Range)
        Shape = len(Group[i].ravel())

        #Colored images
        RGB_Images.append(RGB_Core_Color)
        Shapes.append(Shape)

        #Color Area List
        Red_Density.append(Red)
        Yellow_Density.append(Yellow)
        Green_Density.append(Green)


    return RGB_Images, Red_Density, Yellow_Density, Green_Density, Shapes

def Initial_DMG (Area):
    return ((Area[0]-np.min(Area))*100)

def Total_Recovery (Area):
    return ((np.max(Area)-np.min(Area))*100)

def Recovery_Speed (Area):
    Gradient = list((np.gradient(Area))*100)


    First_Min = list(Area).index(np.min(Area))
    for i in range(len(Gradient[4:])):
        if Gradient[i+4] > 0 and Gradient[i+5] < 0:
            First_Max = i+5
            break
    return (np.mean(Gradient[First_Min:First_Max]))
