import numpy as np

# index_arr = np.array([[2,4],[1,4],[2,4]])

# wh_arr = np.zeros_like(index_arr)
# wh_arr = wh_arr+3
# receptive_field = np.hstack((index_arr,wh_arr))

# print(receptive_field)

# cls = np.array([4,5,6])
# bbx = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])

# mask = np.where(cls>4)

# print(mask)
# print(bbx[mask])

# bbxs = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])

# mask = np.min(bbxs[:,2:],axis=1) >= 4
# bbxs = bbxs[mask]

# print(bbxs)

base = np.array([1,2,1,2])
bbr = np.array([2,3,4,5])

bb = np.multiply(base,bbr)

print(bb)
