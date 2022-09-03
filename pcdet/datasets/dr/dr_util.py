
import sys,io,random, os

if __name__ == '__main__':
    path = '/home/nio/Workspace/dataset/Deeproute_open_dataset/ImageSets'
    os.makedirs(path, exist_ok=True)
    all_index = [str('%05d\n' % i) for i in range(1,10000)]
    random.shuffle(all_index)
    train = all_index[:9000]
    val = all_index[9000:9500]
    test = all_index[9500:]


    with open(path + '/train.txt', 'w') as f:
        f.writelines(train)
        f.close()

    with open(path + '/val.txt', 'w') as f:
        f.writelines(val)
        f.close()

    with open(path + '/test.txt', 'w') as f:
        f.writelines(test)
        f.close()

