import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import box


def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, item in kwargs.items():
        message += f', {key}: {item}'
    sys.stdout.write(message+']')
    sys.stdout.flush()


def makeColorwheel():
    '''
    color encoding scheme
    adapted from the color circle idea described at
    http://members.shaw.ca/quadibloc/other/colint.htm
    '''

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;

    #GC
    colorwheel[col:GC+col, 1]= 255 
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;

    #BM
    colorwheel[col:BM+col, 2]= 255 
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;

    #MR
    colorwheel[col:MR+col, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v) 

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def vis_flow(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    
    u = flow[:,:,0]
    v = flow[:,:,1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0 
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img[:,:,[2,1,0]]

def vis_result(images, image_t_syn, flow, filename = './result.png'):
    '''
    Visualize estimated results
    images: input images to deep-voxel-flow, should be stack of 2 or 3 images shape(2or3, h, w, 3)
    image_t_syn: synthesized image at time slice t, shape(h, w, 3)
    flow: sub-estimated optical flow, shape(h, w, 2)
    '''
    if len(images) == 3:
        num_contents = 6
        is_triplet = True
    elif len(images) == 2:
        num_contents = 4
        is_triplet = False
    else:
        raise ValueError('Invalid number of images')
    
    n_cols = num_contents/2
    tick_config = {'labelbottom':False, 'bottom':False, 'labelleft':False, 'left':False}

    fig = plt.figure(figsize = (8*2, 8*(n_cols/2)))
    
    for i, image in enumerate(images):
        plt.subplot(2, n_cols, i+1)
        plt.imshow(image)
        plt.title(f'Image at time slice {["0", "t", "1"][i]}')
        plt.tick_params(**tick_config)
        plt.xticks([])
        box(False)

    plt.subplot(2, n_cols, n_cols+1)
    plt.imshow(vis_flow(flow))
    plt.title('Estimated optical flow')
    plt.tick_params(**tick_config)
    plt.xticks([])
    box(False)

    plt.subplot(2, n_cols, n_cols+2)
    plt.imshow(image_t_syn)
    plt.title('Synthesized image at time slice t')
    plt.tick_params(**tick_config)
    plt.xticks([])
    box(False)
    
    if is_triplet:
        image_diff = np.mean(np.abs(images[1] - image_t_syn), axis = -1)
        plt.subplot(2, n_cols, n_cols+3)
        plt.imshow(image_diff)
        plt.title('Diff-map between GT and synthesized frame')
        plt.tick_params(**tick_config)
        plt.xticks([])
        box(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()
        
