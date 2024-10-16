from os import listdir, path

sfolders = ["../../../RandLA-Net/data/semantic_kitti/dataset/sequences"]

train = open("train.txt", "w")
val = open("val.txt", "w")
test = open("test.txt", "w")

trainlist = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
vallist = ["08"]
testlist = ["08"] #["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]

for sf in sfolders:
    tr = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in trainlist for f in listdir(path.join(sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(sf, t, 'velodyne', f))]
    v = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in vallist for f in listdir(path.join(sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(sf, t, 'velodyne', f))]
    te = [path.join(sf, t, 'velodyne', f).replace("\\", "/") for t in testlist for f in listdir(path.join(sf, t, 'velodyne')) if not f.startswith('.') and path.isfile(path.join(sf, t, 'velodyne', f))]
    
    for s in tr:
        train.write(s[:-4]+"\n")
    for s in v:
        val.write(s[:-4]+"\n")
    for s in te:
        test.write(s[:-4]+"\n")
    
train.close()
val.close()
test.close()