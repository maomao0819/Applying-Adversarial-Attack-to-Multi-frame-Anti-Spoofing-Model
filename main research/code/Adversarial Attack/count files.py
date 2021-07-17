import os

#path = os.getcwd()    #獲取當前路徑
path = 'adversarial_example_fgsm'

num_dirs = 0 #路徑下資料夾數量
num_files = 0 #路徑下檔案數量(包括資料夾)
num_files_rec = 0 #路徑下檔案數量,包括子資料夾裡的檔案數量，不包括空資料夾


for root, dirs, files in os.walk(path):    #遍歷統計
    for each in files:
        if each[-2:] == '.o':
            print("root", root)
            print("dirs ", dirs)
            print("each ", each)
        num_files_rec += 1
    for name in dirs:
        num_dirs += 1
        print(os.path.join(root, name))

for fn in os.listdir(path):
    num_files += 1
    print("fn ", fn)

print("num_dirs ", num_dirs)
print("num_files ", num_files)
print("num_files_rec ", num_files_rec)
