#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numba
import math


@numba.jit
def trkFNih_small(Tc, K, newispikes0, bspikes, diff, inum0, hnum0):
    trks = np.zeros((inum0, hnum0), dtype=np.float64)
    for idx022 in numba.prange(inum0):
        if newispikes0[idx022] == 1:
            for idx023 in numba.prange(hnum0):
                if bspikes[idx023] == 1:
                    trks[idx022][idx023] = K*math.exp(diff/Tc)
    return trks

@numba.jit
def trkFNih(current_tstep0, Tc, K, newispikes0, hspikes0, spikemem0, inum0, hnum0):
    ihtrks = np.zeros((spikemem0-1, inum0, hnum0) , dtype=np.float64)
    current_idx = current_tstep0%spikemem0
    for idx020 in numba.prange(1, spikemem0):
        idx021 = (current_idx-idx020)%spikemem0
        ihtrks[idx020-1] = trkFNih_small(Tc, K, newispikes0, hspikes0[idx021], idx020, inum0, hnum0)
    return ihtrks

@numba.jit
def trkFNhi_small(Tc, K, newhspikes0, bspikes, diff, inum0, hnum0):
    trks = np.zeros((hnum0, inum0), dtype=np.float64)
    for idx022 in numba.prange(hnum0):
        if newhspikes0[idx022] == 1:
            for idx023 in numba.prange(inum0):
                if bspikes[idx023] == 1:
                    trks[idx022][idx023] = -K*math.exp(diff/Tc)
    return trks

@numba.jit
def trkFNhi(current_tstep0, Tc, K, newhspikes0, ispikes0, spikemem0, inum0, hnum0):
    hitrks = np.zeros((spikemem0-1, inum0, hnum0) , dtype=np.float64)
    hitrks_transposed = np.zeros((spikemem0-1, hnum0, inum0) , dtype=np.float64)
    current_idx = current_tstep0%spikemem0
    for idx020 in numba.prange(1, spikemem0):
        idx021 = (current_idx-idx020)%spikemem0
        hitrks_transposed[idx020-1] = trkFNhi_small(Tc, K, newhspikes0, ispikes0[idx021], idx020, inum0, hnum0)
    for idx24 in range(spikemem0-1):
        for idx25 in range(hnum0):
            for idx26 in range(inum0):
                hitrks[idx24][idx26][idx25] = hitrks_transposed[idx24][idx25][idx26]
    return hitrks



# deleted old trkFNho and trkFNoh and replaced with this

@numba.jit
def trkFNho(current_tstep00, Tc, K, newhspikes00, ospikes00, spikemem00, hnum00, onum00):
    ho_presum = trkFNih(current_tstep00, Tc, K, newhspikes00, ospikes00, spikemem00, hnum00, onum00)
    sum_hotrks = np.zeros((hnum00, onum00), dtype=np.float64)
    count_hotrks = np.zeros((hnum00, onum00), dtype=np.float64)
    for idx220 in range(spikemem00-1):
        for idx221 in numba.prange(hnum00):
            for idx222 in numba.prange(onum00):
                if ho_presum[idx220][idx221][idx222] != 0:
                    sum_hotrks[idx221][idx222] += ho_presum[idx220][idx221][idx222]
                    count_hotrks[idx221][idx222] += 1
    sum_hotrks = sum_hotrks*count_hotrks
    return sum_hotrks



@numba.jit
def trkFNoh(current_tstep00, Tc, K, newospikes00, hspikes00, spikemem00, hnum00, onum00):
    oh_presum = trkFNhi(current_tstep00, Tc, K, newospikes00, hspikes00, spikemem00, hnum00, onum00)
    sum_ohtrks = np.zeros((hnum00, onum00), dtype=np.float64)
    count_ohtrks = np.zeros((hnum00, onum00), dtype=np.float64)
    for idx220 in range(spikemem00-1):
        for idx221 in numba.prange(hnum00):
            for idx222 in numba.prange(onum00):
                if oh_presum[idx220][idx221][idx222] != 0:
                    sum_ohtrks[idx221][idx222] += oh_presum[idx220][idx221][idx222]
                    count_ohtrks[idx221][idx222] += 1
    sum_ohtrks = sum_ohtrks*count_ohtrks
    return sum_ohtrks

# why isnt this ^^^ working???

# def trkFNoh(current_tstep00, Tc, K, newospikes00, hspikes00, spikemem00, hnum00, onum00):
#     sum_ohtrks = trkFNho(current_tstep00, Tc, K, newospikes00, hspikes00, spikemem00, hnum00, onum00)
#     sum_ohtrks = sum_ohtrks*(-1)
#     return sum_ohtrks


# claim for a given timestep in epoch trks, the sum of the trk numerators stays the same
# and the denominator will change by tk_(n-1)-tk_n since the numerator and denominator are always the 
# same for a given time step, we only need to store the sum of the numerators and the count
# so I've just added that into the above fns for ho and oh ACTUALLY I should multiply them right away, they're never needed seperately






# update weights for synapses between i and h
# note should go back and make sure that trkFNih and trkFNhi both return ixh matrix and not hxi
@numba.jit
def hebupdate(weights3, ihtraces3, hitraces3, spikemem3):
    x3 = len(weights3)
    y3 = len(weights3[0])
    newweights = weights3.copy()
    for idx34 in range(spikemem3-1):
        for idx32 in numba.prange(x3):
            for idx33 in numba.prange(y3):
                newweights[idx32][idx33] = weights3[idx32][idx33]*(1+weights3[idx32][idx33]*ihtraces3[idx34][idx32][idx33])*(1+weights3[idx32][idx33]*hitraces3[idx34][idx32][idx33])
    return newweights


# define fn for delta k's, only use this for synapses between H and O
# ughhhhh we're storing traces for 3600 timesteps :(
@numba.jit
def deltak(ktrks, srp, tracediff, c_deltak):
    anum = len(ktrks)
    bnum = len(ktrks[0])
    dkdenom = tracediff+c_deltak
    dk = np.zeros((anum, bnum) , dtype=np.float64)
#     for idx30 in range(anum):#switch to prange after troubleshooting
#         for idx31 in range(bnum):#switch to prange after troubleshooting
    for idx30 in numba.prange(anum):
        for idx31 in numba.prange(bnum):
            dk[idx30][idx31] = srp*ktrks[idx30][idx31]/dkdenom
    return dk
# update weights single k for synapses between h and o
@numba.jit
def supupdate_small(hotargets1, hoItargets1, eweights1, iweights1, srp1, sum_hotrks1, sum_ohtrks1, tracediff1, hnum1, onum1, c_deltak1):
    hotargetsums = np.sum(eweights1, axis=1)
    hoItargetsums = np.sum(iweights1, axis=1)
    dkho = deltak(sum_hotrks1, srp1, tracediff1, c_deltak1)
    dkoh = deltak(sum_ohtrks1, srp1, tracediff1, c_deltak1)
    ecrazy = np.zeros((hnum1, onum1) , dtype=np.float64)
    icrazy = np.zeros((hnum1, onum1) , dtype=np.float64)
#     for idx11 in range(hnum):#switch to prange after troubleshooting
#         for idx12 in range(onum):#switch to prange after troubleshooting
    for idx11 in numba.prange(hnum1):
        for idx12 in numba.prange(onum1):
            ecrazy[idx11][idx12] = (1+hotargets1[idx11][idx12]*dkho[idx11][idx12]/hotargetsums[idx11])*(1+hotargets1[idx11][idx12]*dkoh[idx11][idx12]/hotargetsums[idx11])
            icrazy[idx11][idx12] = (1+hoItargets1[idx11][idx12]*dkho[idx11][idx12]/hoItargetsums[idx11])*(1+hoItargets1[idx11][idx12]*dkoh[idx11][idx12]/hoItargetsums[idx11])
    return ecrazy, icrazy
# update weights ALL k for synapses between h and o
@numba.jit
def supupdate(hotargets1, hoItargets1, eweights1, iweights1, srp1, epoch_hotrks1, epoch_ohtrks1, current_tstep1, epochmem1, hnum1, onum1, c_deltak1):
    enewweights = eweights1.copy()
    inewweights = iweights1.copy()
    ctidx = current_tstep1%epochmem1
#     for idx10 in range(epochmem1): #switch to prange after troubleshooting
    for idx10 in numba.prange(epochmem1):
        if idx10 < ctidx:
            tracediff11 = ctidx -idx10 #this fn assumes sum_trks are stored at k%epochmem
        else:
            tracediff11 = epochmem1-idx10+ctidx
        efactor, ifactor = supupdate_small(hotargets1, hoItargets1, eweights1, iweights1, srp1, epoch_hotrks1[idx10], epoch_ohtrks1[idx10], tracediff11, hnum1, onum1, c_deltak1)
        enewweights *= efactor
        inewweights *= ifactor
    return enewweights, inewweights






#heres a function to initialize the environment
def initenv(environment, rng):
    for itest in range(100):
        x = rng.integers(50)
        y = rng.integers(50)
        check = 0
        for idx in range(-1, 2):
            for idy in range(-1, 3):
                if environment[(x+idx)%50, (y+idy)%50] == 1:
                    check = 1
        if check == 0:
            environment[x,y] = 1
            environment[x,(y+1)%50] = 1
    for itest in range(250):
        x = rng.integers(50)
        y = rng.integers(50)
        check = 0
        for idy in range(-1, 3):
            if idy == -1:
                for idx in range(-1, 2):
                    if environment[(x+idx)%50, (y+idy)%50] == 1:
                        check = 1
            elif idy == 0 or idy == 1:
                for idx in range(-1, 3):
                    if environment[(x+idx)%50, (y+idy)%50] == 1:
                        check = 1
            elif idy == 2:
                for idx in range(3):
                    if environment[(x+idx)%50, (y+idy)%50] == 1:
                        check = 1
        if check == 0:
            environment[x,y] = 1
            environment[(x+1)%50,(y+1)%50] = 1
    return environment


# heres a function from position and the environment to the stimuli that the network receives
@numba.jit
def posenv2stim(position4, environment4, stimsize4, inum4):
    stimuli4 = np.zeros(inum4 , dtype=np.float64)
    x = position4[0]
    y = position4[1]
    halfsize4 = (stimsize4-1)//2
    xi = 0
    stimcount = 0
    for idx40 in range(stimsize4):
        yi = 0
        for idx41 in range(stimsize4):
            xj = (x-halfsize4+xi)%50
            yj = (y-halfsize4+yi)%50
            if xj == x and yj == y:
                yi = (yi+1)%stimsize4
            else:
                stimuli4[stimcount] = environment4[xj, yj]
                yi +=1
                stimcount += 1
        xi +=1
    return stimuli4



# lets write the current update function
@numba.jit
def curupdate(cur4, extcur4, v4, mu4, sigma4):
    newcur = cur4.copy()
    clen = len(cur4)
    for idx42 in numba.prange(clen):
        newcur[idx42] = cur4[idx42] -mu4*(v4[idx42]+1) + mu4*sigma4 + mu4*extcur4[idx42]
    return newcur


# okay lets write that membrane potential function
# @numba.jit
# def vupdate(beta_e5, extcur5, cur5, v5, vpast5, alpha5):
#     vnew = v5.copy()
#     vlen = len(v5)
#     newspikes = np.zeros(vlen , dtype=np.float64)
#     for idx50 in numba.prange(vlen):
#         idxcur = cur5[idx50]+beta_e5*extcur5[idx50]
#         if v5[idx50] <= 0:
#             vnew[idx50] = alpha5/(1-v5[idx50])+idxcur
#         elif v5[idx50] < (alpha5+idxcur) and vpast5[idx50] <= 0:
#             vnew[idx50] = alpha5+idxcur
#         else:
#             vnew[idx50] = -1
#             newspikes[idx50] = 1
#     return vnew, newspikes

# above ^^^ folows text, below is my modification
@numba.jit
def vupdate(beta_e5, extcur5, cur5, v5, vpast5, alpha5):
    vnew = v5.copy()
    vlen = len(v5)
    newspikes = np.zeros(vlen , dtype=np.float64)
    for idx50 in numba.prange(vlen):
        idxcur = cur5[idx50]+beta_e5*extcur5[idx50]
        if v5[idx50] <= 0:
            vnew[idx50] = v5[idx50]+alpha5*(1-v5[idx50])**-1+idxcur
        elif v5[idx50] < (alpha5+idxcur) and vpast5[idx50] <= 0:
            vnew[idx50] = v5[idx50]+alpha5+idxcur
        else:
            vnew[idx50] = -1
            newspikes[idx50] = 1
    return vnew, newspikes


# I can't make sense of how g_syn is supposed to work
# the biggest problem is that since membrane potential can be both positive and negative
# I'm not sure how to make sense of the signage on I^syn. It looks like whenever a post synaptic neuron has
# positive potential, the current would reverse flow. Worse, it seems like this could be stopped by 
# the membrane's reversal potential which is the opposite effect of what the reversal potential value is 
# supposed to have...
# so I'm just goint to use +1 current for excitatory synapses, -1 for inhibitory
# additionally, I'm not going to include actual reversal flow, merely inhibitio (i.e. least possible current = 0)
# note: another oddity is that in 2022 paper X is in [0, 1], but in ref15 in [-1, 1]


# going from i layer spikes to h layers external current
@numba.jit
def hexternal(ispikes6, wih6, hfirefactor6, inum6, hnum6):
    incur6 = ispikes6*hfirefactor6
    hextcur6 = np.zeros(hnum6, dtype=np.float64)
    for idx60 in numba.prange(inum6):
        if incur6[idx60] != 0:
            hextcur6 += incur6[idx60]*wih6[idx60]
    return hextcur6

# going from h layer spikes to o layer external current
@numba.jit
def oexternal(hspikes7, who7, whoI7, ofirefactor7, hnum7, onum7):
    incur7 = hspikes7*ofirefactor7
    oextcur7 = np.zeros(onum7, dtype=np.float64)
    for idx70 in numba.prange(hnum7):
        if incur7[idx70] != 0:
            oextcur7 += incur7[idx70]*who7[idx70]
    for idx71 in numba.prange(hnum7):
        if incur7[idx71] != 0:
            oextcur7 -= incur7[idx71]*whoI7[idx71]
    for idx72 in range(onum7):
        if oextcur7[idx72] < 0:
            oextcur7[idx72] = 0
    return oextcur7



# update network position based on output spikes, but with 1% chance of moving randomly
@numba.jit
def updateposition(output_spikes7, randmove7, randchance7, position7):
    if randchance7 < 0.01:
        move = randmove7
    else:
        move = output_spikes7.argmax()
    if move == 0:
        newposition = np.array([position7[0], position7[1]+1])
    elif move == 1:
        newposition = np.array([position7[0], position7[1]-1])
    elif move == 2:
        newposition = np.array([position7[0]+1, position7[1]])
    elif move == 3:
        newposition = np.array([position7[0]+1, position7[1]+1])
    elif move == 4:
        newposition = np.array([position7[0]+1, position7[1]-1])
    elif move == 5:
        newposition = np.array([position7[0]-1, position7[1]])
    elif move == 6:
        newposition = np.array([position7[0]-1, position7[1]+1])
    elif move == 7:
        newposition = np.array([position7[0]-1, position7[1]-1])
    #lets make sure we stay in the environment
    newposition[0] = newposition[0]%50
    newposition[1] = newposition[1]%50
    return newposition

# determine if reward, punish, or nothing and move food particle if neccessary
@numba.jit
def checkenv(position8, environment8, randfood8, Srp_basic8, Srp_reward8, Srp_punish8):
    px = position8[0]
    py = position8[1]
    if environment8[px][py] == 0:
        newSrp = Srp_basic8
    elif environment8[px+1][py+1] == 1 or environment8[px-1][py-1]:
        newSrp = Srp_punish8
    elif environment8[px][py+1] == 1 or environment8[px][py-1]:
        newSrp = Srp_reward8
        environment8[px][py] = 0
        environment8[px][py+1] = 0
        environment8[px][py-1] = 0
        rfcheck = 0
        for idx80 in range(len(randfood8)):
            rfx = randfood8[idx80][0]
            rfy = randfood8[idx80][1]
            for idx in range(-1, 2):
                for idy in range(-1, 3):
                    if environment8[(rfx+idx)%50, (rfy+idy)%50] == 1:
                        rfcheck = 1
            if rfcheck == 0:
                environment8[rfx,rfy] = 1
                environment8[rfx,(rfy+1)%50] = 1
                rfcheck = 1
    else:
        print('error')
    return newSrp, environment8

#function to normalize (to target rate) the ho and hoI weights and ensure reasonable long term firing rate
@numba.jit
def ho2target(eDtargets8, iDtargets8, who8, whoI8, hnum8, onum8):
    einter = np.sum(who8, axis=0)
    iinter = np.sum(whoI8, axis=0)
    for idx83 in numba.prange(onum8):
        einter[idx83] = (1/einter[idx83])*eDtargets8[idx83]
        iinter[idx83] = (1/iinter[idx83])*iDtargets8[idx83]

    newwho = who8*einter
    newwhoI = whoI8*iinter
    return newwho, newwhoI

#function to remove NaNs from weights
@numba.jit
def nan_neg_correction(wij):
    shape = wij.shape
    wij = wij.ravel()
    wij[np.isnan(wij)] = 0.0000001
    wij = np.maximum(wij, 0.0000001)
    # adding a CAP
    wij = np.minimum(wij, 0.99999)
    wij = wij.reshape(shape)
    return wij
    

@numba.jit
def stdp_epoch(epochlen, current_tstep, epochmem, position, environment, stimsize, inum, hnum, onum, vi, vo, vh, icur, hcur, ocur, 
              beta_e, alpha, K, mu, Tc, sigma, spikemem, ispikes, hspikes, ospikes, ifirefactor, hfirefactor, ofirefactor, eDtargets, iDtargets,
              wih, who, whoI, hotargets, hoItargets, epochSrp, epoch_hotrks, epoch_ohtrks, randfood, Srp_basic, Srp_reward, Srp_punish, 
              randmove, randchance, c_deltak):
    output_spikes = np.zeros(onum, dtype=np.float64)
    for epochidx in range(epochlen):
    #     print('made it this far 1')
        # first thing that happens is I layer receives stimuli
        stimuli = posenv2stim(position, environment, stimsize, inum)
        # update membrane potential for i layer, but first copy potentials to vpast
        vipast = vi.copy()
        vi, newispikes = vupdate(beta_e, ifirefactor*stimuli, icur, vi, vipast, alpha) #note external current is just the stimuli*factor here
        ispikes[current_tstep%spikemem] = newispikes #updating our memory of i layer spikes
        # update i layer current
        iextcur = ifirefactor*stimuli
        icur = curupdate(icur, iextcur, vipast, mu, sigma) #using vpast since membrane potential has already been updated for next time step
        #now use new spikes to calculate external current for h layer
        hextcur = hexternal(newispikes, wih, hfirefactor, inum, hnum)
        # now repeat calculations for h layer
        vhpast = vh.copy()
        vh, newhspikes = vupdate(beta_e, hextcur, hcur, vh, vhpast, alpha)
        hspikes[current_tstep%spikemem] = newhspikes
        hcur = curupdate(hcur, hextcur, vhpast, mu, sigma) #using vpast since membrane potential has already been updated for next time step

        #now use any new spikes to calculate external current for o layer
        oextcur = oexternal(newhspikes, who, whoI, ofirefactor, hnum, onum)
        # calculations for olayer
        vopast = vo.copy()
        vo, newospikes = vupdate(beta_e, oextcur, ocur, vo, vopast, alpha)

        ospikes[current_tstep%spikemem] = newospikes
        ocur = curupdate(ocur, oextcur, vopast, mu, sigma) #using vpast since membrane potential has already been updated for next time step
        # tally o spikes so we know where to move at end of epoch
        output_spikes += newospikes

        # now generate all the traces from spikes
        ihtraces = trkFNih(current_tstep, Tc, K, newispikes, hspikes, spikemem, inum, hnum)
        hitraces = trkFNhi(current_tstep, Tc, K, newhspikes, ispikes, spikemem, inum, hnum)
        # get index for epoch trace storage
        epochk = current_tstep%epochmem

        epoch_hotrks[epochk] = trkFNho(current_tstep, Tc, K, newhspikes, ospikes, spikemem, hnum, onum)
        epoch_ohtrks[epochk] = trkFNoh(current_tstep, Tc, K, newospikes, hspikes, spikemem, hnum, onum)
        # the synapses between i and h have their traces applied immediately
        wih = hebupdate(wih, ihtraces, hitraces, spikemem)
        #shouldn't need this in full code, but here i'll do it manually since we're not in a larger loop
        current_tstep += 1
    # now just move based on output_spikes and update weights for excite and inhibit synampses between h and o
    position = updateposition(output_spikes, randmove, randchance, position)
    # determine if reward, punish, or nothing and move food particle if neccessary
    epochSrp, environment = checkenv(position, environment, randfood, Srp_basic, Srp_reward, Srp_punish)

    # UPDATE weights between h and o
    who, whoI = supupdate(hotargets, hoItargets, who, whoI, epochSrp, epoch_hotrks, epoch_ohtrks, current_tstep, epochmem, hnum, onum, c_deltak)

    # normalizing* the weights between h and o
    who, whoI = ho2target(eDtargets, iDtargets, who, whoI, hnum, onum)
    
    # this is my first attempt to correct for NaNs
    wih = nan_neg_correction(wih)
    who = nan_neg_correction(who)
    whoI = nan_neg_correction(whoI)
    return wih, who, whoI, ispikes, hspikes, ospikes, epoch_hotrks, epoch_ohtrks, output_spikes, position, epochSrp, environment

