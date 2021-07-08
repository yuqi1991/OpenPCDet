
import sys,io,random

if __name__ == '__main__':
    path = '/home/kendrick/Workspace/dataset/Deeproute_open_dataset'
    all_index = [str('%05d\n' % i) for i in range(1,10000)]
    random.shuffle(all_index)
    train = all_index[:8000]
    val = all_index[8000:8999]
    test = all_index[8999:]


    with open(path + '/train.txt', 'w') as f:
        f.writelines(train)
        f.close()

    with open(path + '/val.txt', 'w') as f:
        f.writelines(val)
        f.close()

    with open(path + '/test.txt', 'w') as f:
        f.writelines(test)
        f.close()

