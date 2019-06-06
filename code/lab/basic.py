import numpy as np
import time


# bbxs = np.array([[[2,3,4,5],[1,2,3,4]],[[4,5,6,7],[5,6,7,8]]])

# print(bbxs[:,:,1:5])

# img = np.random.randint(255,size=(10,10))
# print(img)

# print(np.where(img>150))

# bbxs = np.array([[[2,3,4,5],[1,2,3,4]],[[4,5,6,7],[5,6,7,8]]])

# patches = img[bbxs]
# print(patches)

# filters = np.where(bbxs[:,:2] >= 0 and bbxs[:,2:] < [mw,mh])



# x1 = bbxs[:,0]
# y1 = bbxs[:,0]+bbxs[:,2]
# x2 = bbxs[:,1]
# y2 = bbxs[:,1]+bbxs[:,3]
# print(x1)
# print(y1)
# print([1:5])

# filter_patch = img[x1]
# print(np.shape(filter_patch))

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

# base = np.array([1,2,1,2])
# bbr = np.array([2,3,4,5])

# bb = np.multiply(base,bbr)

# print(bb)

# def testlst(ls):
    # ls.append(8)

# a = np.array([1,2,3])
# b = np.tile(a,(1,3))
# print(b)
# b = np.array([1,2,3])
# print(np.dot(a,b))
# print(np.mat(a)*np.mat(b).T)
# print(a*b)
# testlst(a)
# print(a)

# bbxs = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
# mask = np.array([[1,1,1],[2,2,2]])
# print(mask.T)

# bbxs[:,:2] = bbxs[:,:2]+mask.T
# bbxs[:,2:] = bbxs[:,2:]+30

# print(bbxs)

# mask = np.where(bbxs > 3)
# print(mask)

# def test():
    # return []

# print(test())

# print([i for i in range(4)])

# bbxs = np.random.randint(10,size=(5,4))
# landmark = np.random.randint(10,size=(5,10))


# xy0 = bbxs[:,:2]
# wh0 = bbxs[:,2:]

# start = time.time()
# xy = np.tile(xy0,5) #如果不采用这种方式，而是直接采用广播，是否性能更加
# wh = np.tile(wh0,5)
# landmark0 = landmark*wh+xy

# mid = time.time()

# landmark1 = [landmark[:,i*2:i*2+2]*wh0+xy0 for i in range(5)]
# landmark1 = np.concatenate(landmark1,axis=1)

# end = time.time()

# print(mid-start)
# print(end-mid)

# print(landmark0)
# print(landmark1)

# bbxs = np.random.randint(10,size=(5,4))
# bbr = np.random.randint(10,size=(5,4))


# bbxs0 = bbxs.copy()

# start = time.time()
# bbxs0[:,:2] = bbxs0[:,:2]+bbxs0[:,2:]*bbr[:,:2]
# bbxs0[:,2:] = bbxs0[:,2:]+bbxs0[:,2:]*bbr[:,2:]

# mid = time.time()

# wh = bbxs[:,2:]
# wh = np.tile(wh,2)

# bbxs1 = bbr*wh+bbxs

# end = time.time()

# print(mid-start)
# print(end-mid)
# print(bbxs0)
# print(bbxs1)

# cls = np.array([[1,5,9,2],[12,9,11,8]])
# print(cls[::-1])
# mask = np.where(cls > 7) 
# print(mask)

cls = np.array([0.3,0.4,0.5,0.2,0.9,0.4])
cls[[0,1,2,3]] = cls[[2,3,0,1]]
print(cls)
# cls = np.sort(cls)[::-1]
# hard_idx = int(len(cls)*0.7)
# print(hard_idx)
# print(cls[:hard_idx])
