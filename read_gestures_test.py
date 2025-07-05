import numpy as np

def read_database(dir):
    dataset = []
    gesture = 0
    while True:
        path = dir+"/radar_point_cloud_"+str(gesture+1)+".csv"
        
        gesture = gesture + 1
        #print("Open: ", path)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1) #skip header and null point
        except:
            print("Path not found: "+path)
            break
        FrameNumber = 1
        pointlenght = 80 #maximum number of points in array
        framelenght = 80 #maximum number of frames in array
        datalenght = int(len(data))
        gesturedata = np.zeros((framelenght,2,pointlenght))
        counter = 0

        while counter < datalenght:
            #velocity = np.zeros(pointlenght)
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

            framedata = np.array([x_pos,y_pos]) # (2,80)
            
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
    return dataset


def load_data_exp():
    dir = "radar_meas"

    
    labels = [4 for _ in range(10)]

    #Read gestures from chosen directory 
    d = read_database(dir)
    print(labels)
    print(np.shape(d))
    return d , labels


if __name__ == '__main__':
    load_data_exp()
        
    

