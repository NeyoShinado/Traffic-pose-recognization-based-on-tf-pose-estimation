import xlrd
import xlsxwriter
'''
---------------------------------------------------------------------------------------------------------------------------
                                    output human's keypoint with format [x, y] 
                                    head, neck and arms keypoints are accessible
                                    and all of them are normalized and centralized
                                                an example of above is:
[0.44, 0.36	0.47, 0.39	0.44, 0.39	0.44, 0.43	0.49, 0.42	0.50, 0.38	0.54, 0.38	0.56, 0.36	8_155_141_0.png	turn_right]
----------------------------------------------------------------------------------------------------------------------------
'''


def expand_data(item):
    result = []
    for i in range(len(item)):
        try:
            x = float(item[i][:4]) - float(item[1][:4])
            y = float(item[i][6:10]) - float(item[1][6:10])
        except ValueError:
            x = y = None
        result.extend([x, y])
    return result


def data_reshape(src_dir, poses):
    #labels = ["go_straight", "park_right", "stop", "turn_right"]
    target_file = src_dir + "training_data.xlsx"

    # read data
    data = []
    for label in poses:
        source_file = src_dir + label + ".xlsx"
        try:
            wb = xlrd.open_workbook(source_file)
            for sheet in wb.sheets():
                for rownum in range(sheet.nrows):
                    try:
                        fig = sheet.row_values(rownum)[0].split("\\")[-1]
                        item = sheet.row_values(rownum)[1:19]
                        for i in range(len(item)):
                            if item[i] == "":        #* "" term = 0
                                item[i] = "0.00, 0.00"

                        result = expand_data(item)
                        result.extend([fig, label])
                        data.append(result)

                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)


    # write data
    workbook = xlsxwriter.Workbook(target_file)
    worksheet = workbook.add_worksheet()
    for i in range(len(data)):
        for j in range(len(data[i])):
            worksheet.write(i, j, data[i][j])
    workbook.close()


#if __name__ == "__main__":
#    src = "../traffic_pose/keypoint_data/"
#    data_process(src)
