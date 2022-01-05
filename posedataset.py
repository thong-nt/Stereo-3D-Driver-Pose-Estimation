import pandas as pd
import cv2

def Access_data(path):
    df = pd.read_csv(path,header = 0)
    print(df)
    return df

def Get_img(df,indx):
    img_dir = path_to_ds+df['classname'][indx]+"\\"+df['img'][indx]
    #img = cv2.imread(img_dir,cv2.IMREAD_COLOR)
    return img


def Write_data(df,p_header,indx,value):
    df.at[indx,p_header] = value
    print(df)
    

def Save_ds(df):
    df.to_csv('driver_poses.csv', index=False)


#if __name__ == '__main__':
#    df = Access_data(csv_path)
#    Write_data(df)