from shutil import copyfile

#files with images info
files = [
    "/home/andrea/Documents/project/incorrect images/jpeginfo1.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo2.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo3.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo4.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo5.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo6.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo7.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo8.txt",
    "/home/andrea/Documents/project/incorrect images/jpeginfo9.txt",
]
error_pics = []
warning_pics = []

for fname in files:

    with open(fname) as f:
        content = f.readlines()

    #file_path[0]+file_path[1] - use only [0] if the file path is not splitted like in current case
    for c in content:
        if "ERROR" in c:
            file_path = c.split()
            error_pics.append(file_path[0] + " " + file_path[1])
        if "WARNING" in c:
            file_path = c.split()
            warning_pics.append(file_path[0] + " " + file_path[1])

error_list_output = open("/home/andrea/Documents/project/incorrect images/error_pics.txt", 'w')
warning_list_output = open("/home/andrea/Documents/project/incorrect images/warning_pics.txt", 'w')

for p in error_pics:
    error_list_output.write(p + "\n")

error_list_output.close()

for p in warning_pics:
    warning_list_output.write(p + "\n")

warning_list_output.close()

destination_error = "/home/andrea/Documents/project/incorrect images/error/"
destination_warning = "/home/andrea/Documents/project/incorrect images/warning/"

for p in error_pics:
    img_name = p.split("/")
    img_name = img_name[-1]
    copyfile(p, destination_error + img_name)

for p in warning_pics:
    img_name = p.split("/")
    img_name = img_name[-1]
    copyfile(p, destination_warning + img_name)

