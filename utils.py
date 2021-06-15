import os
import glob
def prepare_Paths (BASE_PATH) :
    MONET_PATH = os.path.join(BASE_PATH, "monet_jpg")
    PHOTO_PATH = os.path.join(BASE_PATH, "photo_jpg")
    return MONET_PATH,PHOTO_PATH

def Prepare_list_names(MONET_PATH,PHOTO_PATH) :  # A function to prepare two lists of MONET PHOTO names
    MONET_FILENAMES = sorted(glob.glob(os.path.join(str(MONET_PATH) + '/*.jpg')))
    PHOTO_FILENAMES = sorted(glob.glob(os.path.join(str(PHOTO_PATH) + '/*.jpg')))
    return MONET_FILENAMES,PHOTO_FILENAMES


# base_path = ".\\gan-getting-started\\"
# monet_path,photo_path = prepare_Paths(base_path)
# monet_files,photo_files = Prepare_list_names(monet_path,photo_path)
# print(monet_files,photo_files)