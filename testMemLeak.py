from make_coco_dataset import *
import pdb
import objgraph as og
import gc

a = CocoDataSet()

count = 0
# pdb.set_trace()

# for f, c, i in a:
#     count += 1
#     input('press to continue the test.')
#     if count > 1:
#         print('end condition.')
#         break

a.__iter__()
for s, e in zip(range(0, 200, 100), range(100, 200, 100)):
    input('press to continue the test.')
    temp = a.GetFeats(start_img=s, end_img=e)

input('before del')
gc.collect()

input('wait for end!')
