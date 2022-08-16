import pandas as pd
import os


def change_format():
    data_path = os.getcwd()
    listdirName = data_path+"/Type_1_dealt"
    # finalName = "type_4.csv"
    # dict = {}
    for file in os.listdir(listdirName):
        if file.find("jpg")<0:
            csvname = listdirName+"/"+file+"/model_out.csv"
            num = pd.read_csv(csvname)
            num = num.iloc[0, 1]
            # dict[file] = num
            print("run change_format successful!")
            return num
    # print(dict)
    # dfdict = pd.DataFrame.from_records(dict,index=[0]).T
    # dfdict = dfdict.reset_index()
    # dfdict.columns = ["IMG_ID","GUESS"]
    # print(dfdict)
    # dfdict.to_csv(finalName)

    print("run change_format unsuccessful")