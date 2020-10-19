# import dask.dataframe as dd
#
# from dask.diagnostics import ProgressBar
#
# pbar = ProgressBar()
# pbar.register()
#
# data = dd.read_csv('/Users/peter/Downloads/lg_train/201912_FLD165NBMA_vib_spectrum_modi_train.txt', sep= '\t')
#
#
# num_data = data.shape[0].compute()
# print(num_data)
#
# ng = data["label"] == 1
# data_ng = data[ng]
#
# num_data_ng = data_ng.shape[0].compute()
# print(num_data_ng)



#=============================================================================================================

# ng = 0
# num_line = 1
# # f= open("/Users/peter/Downloads/lg_train/201912_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 0 in 41705
# f= open("/Users/peter/Downloads/lg_train/202001_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 1 in 42171
# # f= open("/Users/peter/Downloads/lg_train/202002_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 0 in 38339
# # f= open("/Users/peter/Downloads/lg_train/202003_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 5 in 33380
# # f= open("/Users/peter/Downloads/lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 44 in 44297
# # f= open("/Users/peter/Downloads/lg_train/202005_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 16 in 48185
# # f= open("/Users/peter/Downloads/lg_train/202006_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 2 in 65792
# # f= open("/Users/peter/Downloads/lg_train/202007_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r') # NG 1 in 39548
#
# # for i in range(4):
# while True:
#     line = f.readline()
#     if not line: break
#     print(num_line)
#     arr = line.split('\t')
#     if arr[0] == '1':
#         ng += 1
#     num_line += 1
#
# f.close()
#
# print("NG : ", ng)

#=============================================================================================================

# import numpy as np
#
# data = np.loadtxt('/Users/peter/Downloads/lg_train/201912_FLD165NBMA_vib_spectrum_modi_train.txt',
#                   delimiter='\t', skiprows=1, usecols=np.arange(4,10004))
#
# print(len(data), data[0])
# print(len(data[0]))



#=============================================================================
# 전체 데이터에서 NG 데이터만을 가져와서 저
# ng = 0
# num_line = 1
#
# ng_file = '/Users/peter/Downloads/ng.txt'
# import glob
# files = glob.glob('/Users/peter/Downloads/lg_train/*.txt')
# data = None
# for idx, file in enumerate(files):
#     print(idx + 1, "번째 파일 ", file, "가져오는 중.")
#     f = open(file, mode='r')
#     f_ng = open(ng_file, mode='a')
#     while True:
#         line = f.readline()
#         if not line: break
#         # print(num_line)
#         arr = line.split('\t')
#         if arr[0] == '1':
#             f_ng.write(line)
#             ng += 1
#             print(num_line, "번째 라인에서 총 ", ng, '번째 ng 확인.')
#         num_line += 1
#
#     f.close()
#
# print("총 ", num_line, "개 데이터 중 총 ", ng, "개 NG 확인.")
#
#
# #=============================================================================
# # 전체 데이터에서 각 stage별 개수 확인
# stages = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
#
# num_line = 1
#
# import glob
# files = glob.glob('/Users/peter/Downloads/lg_train/*.txt')
# data = None
# for idx, file in enumerate(files):
#     print(idx + 1, "번째 파일 ", file, "가져오는 중.")
#     f = open(file, mode='r')
#     while True:
#         line = f.readline()
#         if not line: break
#         # print(num_line)
#         arr = line.split('\t')
#         if arr[2] in stages:
#             if arr[2] in globals():
#                 globals()[arr[2]] += 1
#             else:
#                 globals()[arr[2]] = 1
#         num_line += 1
#         print(num_line)
#
#     f.close()
#
# print("s1 : {}, s2 : {}, s3 : {}, s4 : {}, s5 : {}, s6 : {}, s7 : {}, s8 : {}".format(s1,s2,s3,s4,s5,s6,s7,s8))


# #=============================================================================
# # 4월 데이터에 대해서 스테이지별 분류 및 비정상 데이터와 정상 데이터셋 구성
# stages = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
# num_line = 1
#
# f = open("/Users/peter/Downloads/lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r')
# while True:
#     line = f.readline()
#     if not line: break
#     # print(num_line)
#     arr = line.split('\t')
#     if arr[2] == '1':
#         if arr[2] in globals():
#             globals()[arr[2]] += 1
#         else:
#             globals()[arr[2]] = 1
#     num_line += 1
#     print(num_line)
#
# f.close()
#
# print(globals()[s1])
# for i in range(8):
#     # print('s{}'.format(i+1))
#     print("s", i + 1, " : {}".format(globals()['s{}'.format(i + 1)]))
#     try:
#         print("111")
#         print("s", i+1, " : {}".format( globals()['s{}'.format(i+1)] ) )
#     except:
#         print("222")
#         pass


# print("s1 : {}, s2 : {}, s3 : {}, s4 : {}, s5 : {}, s6 : {}, s7 : {}, s8 : {}".format(s1,s2,s3,s4,s5,s6,s7,s8))


#=============================================================================
# 전체 데이터에서 NG 데이터만을 가져와서 저
ng = 0
num_line = 1
stages = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']

ng_file = '/Users/peter/Downloads/ng_04.txt'
f = open("/Users/peter/Downloads/lg_train/202004_FLD165NBMA_vib_spectrum_modi_train.txt", mode='r')
data = None

f_ng = open(ng_file, mode='a')
while True:
    line = f.readline()
    if not line: break
    # print(num_line)
    arr = line.split('\t')
    if arr[0] == '1':
        f_ng.write(line)
        ng += 1
        print(num_line, "번째 라인에서 총 ", ng, '번째 ng 확인.')
    num_line += 1

    arr = line.split('\t')
    if arr[2] in stages:
        if arr[2] in globals():
            globals()[arr[2]] += 1
        else:
            globals()[arr[2]] = 1
    num_line += 1
    print(num_line)

f.close()

print("총 ", num_line, "개 데이터 중 총 ", ng, "개 NG 확인.")
