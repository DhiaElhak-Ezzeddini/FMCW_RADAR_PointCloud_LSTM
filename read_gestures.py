import numpy as np

def read_database(dir):
    dataset = []
    labels  = []
    gesture = 0
    while True:
        path = "data/" + dir + "/" + dir+"/gesture_"+str(gesture+1)+".csv"
        
        gesture = gesture + 1
        #print("Open: ", path)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=2) #skip header and null point
        except:
            print("Path not found: "+path)
            break
        if "close_fist_horizontally" == dir :
            labels.append(0)
        if "close_fist_perpendicularly" == dir :
            labels.append(1)
        if "hand_to_left" == dir :
            labels.append(2)
        if "hand_to_right" == dir :
            labels.append(3)
        if "hand_rotation_palm_up" == dir :
            labels.append(4)
        if "hand_rotation_palm_down" == dir :
            labels.append(5)
        if "arm_to_left" == dir :
            labels.append(6)
        if "arm_to_right" == dir :
            labels.append(7)
        if "hand_closer" == dir :
            labels.append(8)
        if "hand_away" == dir :
            labels.append(9)
        if "hand_up" == dir :
            labels.append(10)
        if "hand_down" == dir :
            labels.append(11)

        FrameNumber = 1
        pointlenght = 80 #maximum number of points in array
        framelenght = 80 #maximum number of frames in array
        datalenght = int(len(data))
        gesturedata = np.zeros((framelenght,2,pointlenght))
        counter = 0

        while counter < datalenght:
            velocity = np.zeros(pointlenght)
            #peak_val = np.zeros(pointlenght)
            x_pos = np.zeros(pointlenght)
            y_pos = np.zeros(pointlenght)
            iterator = 0

            try:
                while data[counter][0] == FrameNumber:
                    #velocity[iterator] = data[counter][3]
                    #peak_val[iterator] = data[counter][4]
                    x_pos[iterator] = data[counter][5]
                    y_pos[iterator] = data[counter][6]
                    iterator += 1
                    counter += 1
            except:
                print(" ")

            framedata = np.array([x_pos,y_pos]) # (3,80)
            
            try:
                gesturedata[FrameNumber - 1] = framedata
            except:
                print("Frame number out of bound", FrameNumber)
                break

            FrameNumber += 1

        # gesturedata is now of shape (80,3,80) 
        gesturedata = gesturedata.reshape(80,-1) ## ==> reshaping to (80,240) 
        dataset.append(gesturedata)

    print("End of the loop")
    return dataset , labels


def load_data():
    dir = ["close_fist_horizontally", "close_fist_perpendicularly", "hand_to_left", 
        "hand_to_right","hand_rotation_palm_up","hand_rotation_palm_down", 
        "arm_to_left", "arm_to_right","hand_closer", "hand_away", "hand_up", "hand_down"]

    dataset=[]
    labels = []
    #Read gestures from chosen directory
    for i in range(len(dir)) : 
        d , l= read_database(dir[i])
        dataset += d 
        labels+=l
    return dataset , labels


if __name__ == '__main__':
    load_data()
        
    

