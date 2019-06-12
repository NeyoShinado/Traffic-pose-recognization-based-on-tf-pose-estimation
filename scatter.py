import xlrd
import numpy as np
import matplotlib.pyplot as plt


def expand_data(item):
    x_list = []
    y_list = []
    for i in range(len(item)):
        try:
            x = float(item[i][:4]) - float(item[1][:4])
            y = float(item[i][6:10]) - float(item[1][6:10])
        except ValueError:
            x = y = None
        x_list.extend([x])
        y_list.extend([y])
    return x_list, y_list

poses = ["go_straight", "park_right", "stop", "turn_right"]

for pose in poses:
    dir = "../traffic_pose/keypoint_data/" + pose + ".xlsx"
    wb = xlrd.open_workbook(dir)
    sheet = wb.sheets()[0]
    x = []
    y = []
    #x = np.empty((sheet.nrows, 8))
    #y = np.empty((sheet.nrows, 8))
    data = np.empty((sheet.nrows, 8))
    for r in range(sheet.nrows):
        item = sheet.row_values(r)[1:9]
        if "" in item[0:8]:
            continue
        try:
            lx, ly = expand_data(item)
            x.extend(lx)
            y.extend(ly)
        except:
            print("transfer_error")
    print(len(x), len(y))

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    color = ["cornflowerblue", "orangered", "palegreen", "cyan", "gold", "orchid", "brown", "salmon"]
    point = ["head", "neck", "r_shouder", "r_elbow", "r_hand", "l_shouder", "l_elbow", "l_hand"]
    for i in range(8):
        try:
            ax_scatter.scatter(x[i:len(x):8], y[i:len(y):8], c=color[i], label=point[i])
        except:
            print("draw_error")

    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    plt.title(pose)

    fig.savefig('../traffic_pose/keypoint_data/' + pose + '_scatter.jpg')
    plt.show()