import cv2
import numpy as np

def compare_embeddings(emb1t, emb2t):
    emb1 = emb1t.flatten()
    emb2 = emb2t.flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

#def compare_faces(data_emb, emb):

def find_match(data, stored_data, thresh = 0.2):
    ldata = np.linalg.norm(data, axis=1)[None].T
    lstored = np.linalg.norm(stored_data, axis=1)[None].T
    num = np.dot(data, stored_data.T)
    den = np.dot(ldata, lstored.T)
    similarity = num/den
    thresh_vec = np.zeros( (similarity.shape[0],1) ) + thresh
    similarity = np.column_stack(( thresh_vec,similarity ))
    matches = np.argmax(similarity, axis = 1)
    return matches

def true_match(data, stored_data,nnames, unames, thresh = 0.4):
    
    names = nnames.copy()
    names.remove('Uknown')
    matches_t = find_match(data, stored_data, thresh)

    names = np.asarray(names)
    #unique_names = np.unique(names)
    unique_names = unames
    t_match = np.ones( (matches_t.shape[0], 1) )

    for name in unique_names:
        un_ind = np.where(names == name)[0]
        nmax,nmin = np.max(un_ind),np.min(un_ind)
        name_matches = matches_t[:,nmin:nmax + 1]
        t_match = np.column_stack(( t_match,np.sum(name_matches,axis=1)[:,None]))
    
    r_match = np.argmax(t_match, axis= 1)


def scaller_dist(img):
    dim = (112,112)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def scaller_conc(img):
    oheight,owidth,_ = img.shape
    if(oheight >= owidth):
        height = 112
        p = (height/oheight)
        width = owidth*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        nl = int((112 - resized.shape[1] )/2)
        lines = np.zeros( (resized.shape[0],nl,3), np.uint8  )
        resized = np.column_stack(( lines,np.column_stack(( resized, lines )) ))
        if( ((112 - resized.shape[1])/2) != 0):
            resized = np.column_stack(( np.zeros((resized.shape[0],1,3),np.uint8 ), resized ) )
        return resized

    elif(owidth > oheight):
        width = 112
        p = (width/owidth)
        height = oheight*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        nl = int((112 - resized.shape[0] )/2)
        lines = np.zeros( (nl,resized.shape[1],3),np.uint8  )

        resized = np.row_stack(( lines,np.row_stack(( resized, lines )) ))
        if( ((112 - resized.shape[0])/2) != 0 ):

            resized = np.row_stack(( np.zeros((1,resized.shape[1],3) ,np.uint8), resized ) )
        return resized

    else:
        return img


def scaller(img):
    oheight,owidth,_ = img.shape
    if(oheight >= owidth):
        width = 112
        p = (width/owidth)
        height = oheight*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        res = height - 112

        resized = resized[ int(res/2):(int(res/2) + 112), :,:]
        if(resized.shape != (112,112,3) ):
            print('============ERROR WITH SCALLER 2 ============')

        return resized
    elif(owidth > oheight):
        print('============ERROR WITH SCALLER 1 ============')
        #print(oheight)
        height = 112
        p = (height/oheight)
        width = owidth*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        res = width - 112

        resized = resized[ :, int(res/2):(int(res/2) + 112),:]
        if(resized.shape != (112,112,3) ):
            print('============ERROR WITH SCALLER 2 ============')

        return resized
    else:
        return