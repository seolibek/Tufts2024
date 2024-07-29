# Curr goal: write LUND algorithm based on proposed algorithm in LUND paper.
# With a working LUND scheme, incorporate autoencoder model into here:
# (assuming) my autoencoder can learn some accurate representation of my data,
# use the latent representation of the lower dim data to build some markov 
# transition matrix P.
# Then, go through entire scehma, which outputs the cluster assignments for the 
# lower dimensional data. 
# Afterwards, take cluster labeling, compute some centroid of points assigned to that cluster
# in the latent space. Then, use decoder to decode the centroids back to high dim space.

