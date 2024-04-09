from os import listdir, path

rootp = "../../../RandLA-Net/data/semantic_kitti/"
sfolders = ["dataset/sequences"]

method = "CIL"

train = open(method + "/step_0.txt", "w")
val = open(method + "/step_1.txt", "w")
test = open(method + "/step_2.txt", "w")

# suddivisione equa per punti
step_0 = ["01", "02", "03"]
step_1 = ["04", "05", "09", "10"]
step_2 = ["00", "06", "07"]

# suddivisione equa per classe
#step_0 = ["01", "03", "04", "07", "09"]
#step_1 = ["05", "06", "10"]
#step_2 = ["00", "02"]

for sf in sfolders:
    tr = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in step_0 for f in listdir(path.join(rootp, sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(rootp, sf, t, 'velodyne', f))]
    v = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in step_1 for f in listdir(path.join(rootp, sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(rootp, sf, t, 'velodyne', f))]
    te = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in step_2 for f in listdir(path.join(rootp, sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(rootp, sf, t, 'velodyne', f))]
    
    for s in tr:
        train.write(s[:-4]+"\n")
    for s in v:
        val.write(s[:-4]+"\n")
    for s in te:
        test.write(s[:-4]+"\n")
    
train.close()
val.close()
test.close()