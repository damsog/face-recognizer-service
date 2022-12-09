import cv2
import numpy as np

def compare_embeddings(emb1t, emb2t):
    emb1 = emb1t.flatten()
    emb2 = emb2t.flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

# This function is used to compare the embeddings of the face against the stored embeddings.
# We use cosine similarity to compare the embeddings.
# For optimization, we calculate the cosine simmilarity using lineal algebra to compute it faster.
# For more info about the cosine similarity, check this link: https://en.wikipedia.org/wiki/Cosine_similarity
def find_match(vector, stored_vectors, thresh = 0.5):
    # Euclidean norm of the input vector and the stored vectors
    norm_vector = np.linalg.norm(vector, axis=1)[None].T
    norm_stored_vectors = np.linalg.norm(stored_vectors, axis=1)[None].T
    # Dot product of the input vector and the stored vectors. fast way to compute the dot product
    # of the vector against all the stored vectors
    num = np.dot(vector, stored_vectors.T)
    # Dot product of the norms of the input vector and the stored vectors. fast way to compute the dot product
    # of the norms of the vector against all the stored vectors
    den = np.dot(norm_vector, norm_stored_vectors.T)
    # Cosine similarity
    similarity = num/den

    # We return the index of the stored vectors that have a similarity higher than the threshold
    matches = np.where(similarity>thresh, True, False)
    return matches

# This function is used to compare the embeddings of the face against the stored embeddings.
# After comparing them, we return the label of the person that has the highest similarity, if any.
# Each row of embeddings corresponds to an incoming embedding (i.e. each face)
def true_match(embeddings, stored_embeddings,labels, unique_labels, thresh = 0.4):
    # We create an array of zeros to store the number of times each label appears in the embedding_matches array
    # Each row corresponds to an incoming embedding (i.e. each face) and each column corresponds to a label
    # The first column corresponds to the label 'Unknown', later, for each label, we add a column
    labels_count = np.zeros( (embeddings.shape[0], 1) , dtype=int)

    # Getting the embeddings that match with the stored embeddings. we obtain an array of booleans
    embedding_matches = find_match(embeddings, stored_embeddings, thresh)

    labels = np.asarray(labels)

    # We need to count how many times each label appears in the embedding_matches array
    # We iterate over each unique label, but not the unknown label which will only matter if every label sums 0
    for name in unique_labels:
        # Slice the embedding_matches for the subarray region corresponding to the name
        name_limits = np.where(labels == name)[0]
        # We decrease the limits by 1 because the labels contain the unknown label as the first element
        upper_name_limits,lower_name_limits = np.max(name_limits) - 1,np.min(name_limits) - 1
        embedding_matches_name = embedding_matches[:,lower_name_limits:upper_name_limits + 1]
        
        # Count how many times the name appears (its true) in the embedding_matches_name array
        # This already counts for each incoming embedding (i.e. each face)
        labels_count = np.column_stack(( labels_count,np.sum(embedding_matches_name,axis=1)[:,None]))

    # We get the index of the label that appears the most times in the embedding_matches array
    label_match = np.argmax(labels_count, axis= 1)

    return label_match

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