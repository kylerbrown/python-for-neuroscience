
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as spm

def image_filter(img_path, timefunc=lambda x, t: x - x*np.exp(-t), tlen=50,
                 img_size=(10, 10), incremt=.1):
    img = spm.imread(img_path, flatten=True)
    img_small = spm.imresize(img, img_size)
    imgmean = np.mean(img_small[img_small > 1])
    img_small[img_small < 40] = imgmean
    img_norm = (img_small - img_small.mean())/(1.5*img_small.std())
    filt = np.zeros((tlen,)+img_size)
    for t in xrange(tlen):
        mod = timefunc(1, t*incremt)
        filt[t, :, :] = mod*img_norm
    return filt

def create_filter(n_ms, func, dom, ms_step=1.):
    nums = np.linspace(dom[0], dom[-1], n_ms/ms_step)
    filt = func(nums)
    return filt

def plot_1dfilter(filt, rel_ms, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    xs = np.linspace(rel_ms[0], rel_ms[1], len(filt))
    ax.plot(xs, filt)
    plt.show(block=False)
    return ax

def gen_binary_stim(n, binvals=(-1, 1)):
    stimtrain = np.random.random_sample(n)
    stimtrain[stimtrain < .5] = binvals[0]
    stimtrain[stimtrain >= .5] = binvals[1]
    return stimtrain

def gen_gaussian_stim(n, mu=0, sigma=1.):
    stimtrain = sigma*np.random.standard_normal(n) + mu
    return stimtrain

def apply_filter_to_stim(filt, stimtrain, avg_rate=20., var=20.):
    if len(filt.shape) == 1:
        conv = np.convolve(stimtrain, filt[::-1], 'valid')
    elif len(filt.shape) == 3:
        conv = np.zeros(len(stimtrain) - len(filt) + 1)
        for i in xrange(len(conv)):
            conv[i] = np.sum(stimtrain[i:i+len(filt), :, :]*filt)
    conv_norm = (conv - conv.mean()) / conv.std()
    avg_rate = avg_rate/1000.
    scale = var/1000.
    conv_sps = (conv_norm + avg_rate)*np.sqrt(scale)
    conv_sps[conv_sps < 0] = 0
    spks = conv_sps >= np.random.rand(len(conv_sps))
    return np.where(spks == 1)[0] + len(filt)

def make_nd_sta(stim, spks, pre=100, post=10):
    dim = (len(spks), pre+post) + stim.shape[1:]
    allsts = np.zeros(dim)
    for i, s in enumerate(spks):
        if s < len(stim) - post and s >= pre:
            allsts[i] = stim[s-pre:s+post]
        else:
            allsts[i] = np.nan
    sta = np.nanmean(allsts, 0)
    return sta

def make_sta(stim, spks, pre=250, post=50):
    allsts = np.zeros((len(spks), pre+post))
    for i, s in enumerate(spks):
        if s < len(stim) - post and s >= pre:
            allsts[i, :] = stim[s-pre:s+post]
        else:
            allsts[i, :] = np.nan
    sta = np.nanmean(allsts, 0)
    return sta

def plot_3d_sta(sta, stapre, stapost, n_ts=7.):
    f = plt.figure()
    slices_t = np.floor(sta.shape[0]/float(n_ts))
    n = np.ceil(np.sqrt(n_ts))
    allmin = np.min(sta)
    allmax = np.max(sta)
    ts = np.arange(-stapre, stapost)
    for i in xrange(int(n_ts)):
        sli = np.mean(sta[i*slices_t:(i+1)*slices_t], 0)
        ax = f.add_subplot(n, n, i+1)
        ax.imshow(sli, vmin=allmin, vmax=allmax)
        t0, t1 = ts[i*slices_t], ts[(i+1)*slices_t - 1]
        ax.set_title('{} to {}'.format(t0, t1))
    f.tight_layout()
    plt.show(block=False)

def make_filt_sta(stimlen, stapre, stapost, filtfunc=np.sin, filtlen=50, 
                  dom=(0, 2*np.pi), binary_stim=True, plot=False,
                  filt_img=None, incremt=.1):
    if filt_img is None:
        filt = create_filter(filtlen, filtfunc, dom)
    else:
        filt = image_filter(filt_img, tlen=filtlen, incremt=incremt)
        stimlen = (stimlen,) + filt.shape[1:]
    if binary_stim:
        stim = gen_binary_stim(stimlen)
    else:
        stim = gen_gaussian_stim(stimlen)
    spks = apply_filter_to_stim(filt, stim)
    sta = make_nd_sta(stim, spks, pre=stapre, post=stapost)
    if plot:
        f = plt.figure()
        ax_filt = f.add_subplot(1, 1, 1)
        filt_xs = np.linspace(-filtlen, 0, len(filt))
        sta_xs = np.linspace(-stapre, stapost, len(sta))
        ax_filt.plot(filt_xs, filt, label='true filter')
        ax_filt.plot(sta_xs, sta, label='sta')
        plt.show(block=False)
    return sta, spks, stim
    
    
