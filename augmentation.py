import glob
import numpy as np
from imblearn.over_sampling import SMOTE


class Augmentation(object):
    def __init__(self):
        self.file_augmented = '../augmented.txt'
        self.files = glob.glob('../lg_train/*.txt')
        self.file_ng = '../ng.txt'
        self.data_freq = []
        self.data_label = []
        

    def load_data(self, num):
        # get data of lable 0 same number as input parameter num
        cnt = 0
        
        for idx, file in enumerate(self.files):
            # print(idx + 1, "번째 파일 ", file, "가져오는 중.")
            f = open(file, mode='r')
            while True:
                line = f.readline()
                if cnt > num: break
                if not line: break
                # print(num_line)
                arr = line.strip().split('\t')
                if arr[0] == '0': # if label 0, 
                    self.data_freq.append(arr[4:])
                    self.data_label.append(arr[0])
                    # print(len(arr[4:]))
                    # print(arr[-1])
                cnt += 1
            f.close()
        print(f'{num} label 0 data loaded.')

    def augment(self, num):
        self.load_data(num)
        # append ng data to data list
        cnt = 0
        f = open(self.file_ng, mode='r')
        while True:
            line = f.readline()
            if not line: break
            arr = line.strip().split('\t')

            self.data_freq.append(arr[4:]) # 0Hz to 10000Hz
            self.data_label.append(arr[0]) # label( 0 or 1 )
            cnt += 1
        f.close()
        print(f'{cnt} label 1 data loaded.')
        print(f'total {len(self.data_label)} data in data list.')

        self.data_freq = np.array(self.data_freq).astype(np.float64) # change dtype float 64 for SMOTE
        self.data_label = np.array(self.data_label).astype(np.int)
        
        # augment data with SMOTE
        sm = SMOTE(random_state=0)
        freq_smote, label_smote = sm.fit_sample(self.data_freq, self.data_label)
        
        print(f'augment complete. total {len(label_smote)} datas.')

        # save augmented data to txt file
        f = open(self.file_augmented, mode='a')
        
        for label, freq in zip(self.data_label, self.data_freq) :
            line = str(label) + '\t' + '\t'.join(map(str,freq)) + '\n'
            f.write(line)

        f.close()


def main():
    au = Augmentation()
    au.augment(100)

if __name__ == "__main__":
    main()