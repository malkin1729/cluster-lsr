import torch as T

def update_b(qis, qib, qib_other, pbb, qic, n_iter, no_psb, no_pbb, no_pbc, window, scale):

    qis_blur = bag(qis,window)

    pb = qib.mean((1,2))

    psb = T.einsum('bxy,sxy->sb',qib,qis_blur)
    psb /= psb.sum(0)
    psb += 0.0001
    
    qic_blur = bag(qic,window)
    
    pbc = T.einsum('bxy,cxy->bc',qib,qic_blur)

    if no_psb: psb[:] = qis.mean((1,2)).unsqueeze(1)**(1/scale)

    qib = (bag(T.einsum('sb,sxy->bxy',T.log(psb),qis),window)*scale).softmax(0)
    
    if not no_pbc: qib *= (bag(T.einsum('bc,cxy->bxy',T.log(pbc),qic),window)).softmax(0)
    
    if no_pbb: 
        qib = T.einsum('b,bxy->bxy',pb,qib)
    else:
        qib *= (T.einsum('to,oxy->txy', T.log(pbb), qib_other)).softmax(0)
        
    qib /= qib.sum(0)
    qib+=0.0001

    return pb, psb, qib, pbc

def update_b_pair(qis1, qib1, qis2, qib2, qic1, qic2, n_iter, no_psb, no_pbb, no_pbc, window, scale):
    
    for _ in range(n_iter):
        
        pbb = T.einsum('axy,bxy->ab', qib1, qib2)
        
        pb1, psb1, qib1, pbc1 = update_b(qis1, qib1, qib2, pbb, qic1, n_iter, no_psb, no_pbb, no_pbc, window, scale)
        pb2, psb2, qib2, pbc2 = update_b(qis2, qib2, qib1, pbb.t(), qic2, n_iter, no_psb, no_pbb, no_pbc, window, scale)
       
    return pb1, psb1, qib1, pb2, psb2, qib2, pbc1, pbc2

def update_s(data, qis, pcs, qic, psb, qib, n_iter, no_pcs, renorm, window, scale):
    
    for _ in range(n_iter):
        mxs, vxs = mean_var(data, qis)

        f = T.einsum('sb,bxy->sxy',T.log(psb),qib)
        qis = ( bag(f,window)*scale + \
                T.einsum('cs,cxy->sxy',T.log(pcs),qic) + \
                lp_gaussian(data,mxs,vxs)*scale ).softmax(0)
        qis[T.isnan(qis)]=0
        qis += 0.0001
        
    pcs = T.einsum('sxy,cxy->cs',qis,qic)
    if renorm: pcs /= pcs.sum(1).unsqueeze(1)
    
    pcs += 0.0001
    
    if no_pcs: pcs[:] = 1.
        
    return qis, pcs

def update_c(pcs, qis, pic, qib, pbc, no_pbc, window):
    
    qic = (T.einsum('cs,sxy->cxy',T.log(pcs),qis)).softmax(0) * pic
    
    if not no_pbc:
        f = T.einsum('bc,bxy->cxy',T.log(pbc),qib)
        qic *= (bag(f,window)).softmax(0)
        
    qic /= qic.sum(0)
    qic+=0.0001
    return qic

def mean_var(data, weight, var_min=0.000001, mq_min=0.000001):
    mq = weight.mean((1,2)).clamp(min=mq_min)
    mean = T.einsum('zij,cij->czij', data, weight).mean((2,3)) / mq.unsqueeze(1)
    var = T.einsum('zij,cij->czij', data**2, weight).mean((2,3)) / mq.unsqueeze(1) - mean**2
    var.clamp_(min=var_min)
    return mean, var

def lp_gaussian(data, mean, var):
    m0 = -1 / var
    m1 = 2 * mean / var
    m2 = -mean**2 / var
    L = T.log(var)
    return (T.einsum('cz,zij->cij',m0,data**2) + T.einsum('cz,zij->cij',m1,data) + (m2-L).sum(1).view(-1,1,1)) / 2

def bag(p, radius):
    return T.nn.functional.avg_pool2d(p.unsqueeze(0), 2*radius+1, stride=1, padding=radius, count_include_pad=False)[0]