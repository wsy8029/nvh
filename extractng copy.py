# -*- coding: utf-8 -*-
# +
num_line = 1
degc_max = 60.0
degc_min = 60.0
eda_file = '../EDA.txt'
import glob
files = glob.glob('../lg_train/*.txt')
data = None
for idx, file in enumerate(files):
    print(idx + 1, "번째 파일 ", file, "가져오는 중.")
    f = open(file, mode='r')
    while True:
        header = f.readline()
        line = f.readline()
        if not line: break
        # print(num_line)
        arr = line.split('\t')
        degc = float(arr[3])
        if degc > degc_max:
            degc_max = degc
        elif degc < degc_min:
            degc_min = degc
        else:
            pass
        num_line += 1
    print(f'{file} 파일까지 {num_line} 라인 확인')

    f.close()
f_eda = open(eda_file, mode='a')
f_eda.write(f'degc_max : {degc_max}')
f_eda.write(f'degc_min : {degc_min}')
f_eda.close()
print("총 ", num_line, "개 데이터 확인.")

# -




