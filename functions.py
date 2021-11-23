import numpy as np
import time, math


def reconstructed_phase_space(observations, tau=15, m=3):
    '''
    Obtain reconstructed phase space with delayed coordinates using 1-D timeseries observations
    
    observations: 1D timeseries
    tau: delay (number of timesteps
    m: dimension of reconstructed phase space
    '''
    x_t = observations
    T = len(x_t)
    yd = np.zeros((T-(m-1)*tau,m))
    for i in range(m):
        yd[:,i] = x_t[i*tau:T-(m-i-1)*tau]
        
    return yd


def closest_point(pointid, pointsAll, dmin=0):
    '''
    Closest point in phase space to a given point
    '''
    point = pointsAll[pointid, :]
    dist = np.linalg.norm(pointsAll-point, axis=1)
    feasible_locs = np.argwhere(dist>dmin).reshape((-1,)) #strictly greater distance, so the point itself is avoided
    sorted_dist = np.argsort(dist[feasible_locs])
    closest_feasible = feasible_locs[sorted_dist[0]]
    return closest_feasible
    

def fixed_time_evol(numsteps, fidloc, clos, yd):
    '''
    Fixed time evolution algorithm: allow separation distance to evolve for fixed number of timesteps
    '''
    fidloc_new = fidloc-1 #only for the initial condition in the following loop, no significance otherwise
    clos_new = clos-1
    lengths = []

    for i in range(numsteps+1): #first step computes initial distance
        fidloc_new += 1
        clos_new += 1
        newLen = np.linalg.norm(yd[fidloc_new, :]-yd[clos_new,:])
        lengths.append(newLen)
        
    
    return fidloc_new, clos_new, lengths


def presOrient(vectorInit, pos, pointsAll, len_thr=5,lmin=0.1, minimize='angle', angle_thr= 1/3, tevol=30):
    '''
    Choose replacement point to preserve orientations and minimize length
    
    pos: current fiducial point index
    vectorInit: the reference vector with which the angle must be minimized
    pointsAll: all points in reconstructed phase space
    '''
    if minimize=='angle':
        except_pos = [i for i in range(pointsAll.shape[0]) if i!=pos] 
        diffVec = pointsAll[except_pos,:]-pointsAll[pos,:] #so all indices after pos have new_ind=ind-1
        cosAll = np.dot(diffVec, vectorInit.reshape((-1,1))).flatten()/(np.linalg.norm(diffVec, axis=1)*np.linalg.norm(vectorInit))
        anglesAll = np.arccos(cosAll)
        anglesSortedInd = np.argsort(anglesAll) #indices which sort angles in ascending order
        lengths_angleSorted = np.linalg.norm(diffVec[anglesSortedInd, :], axis=1)
        suitable_len_ind = np.sort(np.argwhere((lengths_angleSorted)<len_thr)) #first location where length is acceptable
        replacement_pt = anglesSortedInd[suitable_len_ind[0]]

        if replacement_pt>=pos: #was after pos, so actual index = index+1
            replacement_pt+= 1
        i = 1
        while replacement_pt>= pointsAll.shape[0]-tevol:
            replacement_pt = anglesSortedInd[suitable_len_ind[i]]
            i+=1

        return replacement_pt
    
    elif minimize=='len':
        except_pos = [i for i in range(pointsAll.shape[0]) if i!=pos] 
        diffVec = pointsAll[except_pos,:]-pointsAll[pos,:] #so all indices after pos have new_ind=ind-1
        lenAll = np.linalg.norm(diffVec, axis=1)
        cosAll = np.dot(diffVec, vectorInit.reshape((-1,1))).flatten()/(lenAll*np.linalg.norm(vectorInit))
        anglesAll = np.arccos(cosAll)
        
        feasible_ind = np.argwhere( (lenAll>lmin) & (anglesAll<angle_thr)).reshape((-1,))
        if len(feasible_ind)==0:
            print(f"Error: No feasible points. Please change the angle threshold (upper) and/or the length threshold (lower)")
            print(f"smallest angle available={np.amin(anglesAll[lenAll>lmin])}, angle_threshold={angle_thr}")
        else:
            feasible_ind = feasible_ind[feasible_ind<pointsAll.shape[0]-tevol-1] #allow fixed evolution for 1 iteration
            sorted_len_ind = np.argsort(lenAll[feasible_ind]).reshape((-1,))
            replacement_pt = feasible_ind[sorted_len_ind][0]
            if replacement_pt>=pos: #was after pos, so actual index = index+1
                replacement_pt+= 1
        return replacement_pt
    else:
        print(f"invalid input for argument 'minimize'. Valid inputs: 'len' or 'angle'")
        

def compute_lambda(yd, t, fidloc_init, tevol, dmin, anglemax):
    """
    Computes and returns the largest lyapunov exponent, along with:
    local lyapunov exponent
    lengths of separation vectors in each fixed-time evolution cycle
    initial and final lengths in each fixed-time evolution cycle
    """
    lengthsInit = [] #just after replacement- initial length
    lengthsFinal = [] #just before replacement- end of one cycle
    lengthsAll = []
    N, m = yd.shape
    fidloc = fidloc_init
    clos = closest_point(fidloc, yd, dmin=dmin)
    niter = math.floor((N-fidloc)/tevol)-1 #number of replacements
    
    tic = time.time()
    for j in range(niter):
        vectorInit = yd[clos,:]-yd[fidloc,:]
        fidloc, _, lengths = fixed_time_evol(tevol, fidloc, clos, yd)
        lengthsInit.append(lengths[0])
        lengthsFinal.append(lengths[-1])
        lengthsAll.append(lengths)
        clos = presOrient(vectorInit=vectorInit, pos=fidloc, pointsAll=yd, lmin=dmin, minimize='len', angle_thr=anglemax, tevol=tevol)
    # print(f"for {niter} iterations (fixed time evolution cycles), time taken={time.time()-tic}s")
    
    log_ratio = np.log2( np.array(lengthsFinal)/np.array(lengthsInit) )
    local_lam = log_ratio/(t[tevol]-t[0])
    
    lamAll = []
    for i in range(niter):
        lam = 0
        if i>0:
            lam = np.sum(log_ratio[:i])/(t[i*tevol]-t[0])
        lamAll.append(lam)
    
    return lamAll, local_lam, lengthsAll, lengthsInit, lengthsFinal