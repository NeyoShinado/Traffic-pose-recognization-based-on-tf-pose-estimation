from PIL import Image
import os.path
import glob
import time


def resize_jpg(file, outdir):
    try:
        box = Image.open(r"..\traffic_pose\black.png")
        box = box.resize((500, 500), Image.BILINEAR)
        img = Image.open(file)
        wim, him = img.size
        wb, hb = box.size
        box.paste(img, ((wb-wim)//2, (hb-him)//2))
        box.save(os.path.join(outdir, os.path.basename(file)))

    except Exception as e:
        print(e)


def resize_data(data_folder):

    #for p in data_folder:
    #
    #    files = glob.glob(r"../traffic_pose/%s/*.jpg" %p)
    #    for i, file in enumerate(files):
    #        t = time.time()
    #        resize_jpg(file, r"../traffic_pose/%s_new" %p)
    #        elapse = time.time() - t
    #        print("Resize image #%d in %s second" %(i, elapse))
    files = glob.glob(data_folder + "*.jpg")
    for i, file in enumerate(files):
        t = time.time()
        resize_jpg(file, "..\\traffic_pose\\src_resize")
        elapse = time.time() - t
        print("Resize image #%d in %s second" %(i, elapse))