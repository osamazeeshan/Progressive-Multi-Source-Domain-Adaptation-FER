import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min

def create_relv_src_dic(srcs_sub_loader, tar_sub_loader, model, relv_feat_dic, relv_sample_dic, relv_label_dic, relv_sample_count, device):
    model.eval()

    concat_tar_features = []
    concat_src_features = torch.tensor(relv_feat_dic.tolist()).to(device) if len(relv_feat_dic) > 0 else []
    relv_src_data = relv_sample_dic.tolist() if len(relv_sample_dic) > 0 else []
    relv_src_label = relv_label_dic.tolist() if len(relv_label_dic) > 0 else []
    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_label = sources
            src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
            src_feat = model.forward_features(src_data)
            concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
            # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data
            relv_src_data.extend(src_data.detach().cpu().numpy())
            relv_src_label.extend(src_label.detach().cpu().numpy())

            tar_data, _ = target
            tar_data = tar_data.cuda().float()
            tar_feat = model.forward_features(tar_data)
            concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

    mean_feature = torch.mean(concat_tar_features, dim=0)
    src_dist = torch.norm(concat_src_features - mean_feature, dim=1)
    sorted_src_dist, sorted_src_indices = torch.sort(src_dist)
    relv_indices = sorted_src_indices[:relv_sample_count]

    # include current samples into a relv dic
    # relv_sample_dic = torch.cat([relv_sample_dic, src_dist], dim=0) if len(src_dist) > 0 else src_dist

    relv_src_data_arr = np.array(relv_src_data)
    relv_src_data_arr = relv_src_data_arr[relv_indices.tolist()]
    relv_src_label_arr = np.array(relv_src_label)
    relv_src_label_arr = relv_src_label_arr[relv_indices.tolist()]

    relv_feat_arr = np.array(concat_src_features.cpu().numpy())
    relv_feat_arr = relv_feat_arr[relv_indices.tolist()]

    # relv_src_samples = {i.item(): relv_src_data[i].cpu().numpy() for i in relv_indices}

    return relv_feat_arr, relv_src_data_arr, relv_src_label_arr

def create_relv_src_clusters(srcs_sub_loader, tar_sub_loader, model, relv_feat_dic, relv_sample_dic, relv_label_dic, relv_sample_count, tar_name, device, is_before_adapt=False):
    model.eval()

    concat_tar_features = []
    curr_src_features = []
    prev_data_feat = []
    concat_src_features = torch.tensor(relv_feat_dic.tolist()).to(device) if len(relv_feat_dic) > 0 else []
    relv_src_data = relv_sample_dic.tolist() if len(relv_sample_dic) > 0 else []
    relv_src_label = relv_label_dic.tolist() if len(relv_label_dic) > 0 else []

    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_label = sources
            src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
            src_feat = model.forward_features(src_data)
            curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 
            concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
            # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data
            relv_src_data.extend(src_data.detach().cpu().numpy())
            relv_src_label.extend(src_label.detach().cpu().numpy())

            # prev_data = model.forward_features(prev_data_iter)
            # prev_data_feat = torch.cat([prev_data_feat, prev_data], dim=0) if len(prev_data_feat) > 0 else prev_data 

            tar_data, _ = target
            tar_data = tar_data.cuda().float()
            tar_feat = model.forward_features(tar_data)
            concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

    # Define the number of clusters
    n_clusters = 2

    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
    concat_tar_features = concat_tar_features.cpu().numpy()
    concat_src_features = concat_src_features.cpu().numpy() # combining all the relevant + new src sb
    curr_src_features = curr_src_features.cpu().numpy()

    prev_src_feat = relv_feat_dic if len(relv_feat_dic) > 0 else curr_src_features[:relv_sample_count]

    # Fit the model and get the cluster labels
    kmeans.fit(concat_tar_features)
    labels = kmeans.labels_
    # Get the cluster centers
    centroids = kmeans.cluster_centers_

    print("\nTarget Centroid:", centroids)


    # Calculate/Re-calculating the distances of relv src dic with the target cluster centroid
    closest_centroids, distances = pairwise_distances_argmin_min(concat_src_features, centroids)
    # Sort distances and get feature indices
    sorted_indices = np.argsort(distances)
    # get the sorted top N previous src samples indices
    relv_indices = sorted_indices[:relv_sample_count].tolist()
    relv_src_data_arr = np.array(relv_src_data)
    relv_src_data_arr = relv_src_data_arr[relv_indices]
    relv_src_label_arr = np.array(relv_src_label)
    relv_src_label_arr = relv_src_label_arr[relv_indices]

    relv_feat_arr = np.array(concat_src_features)
    relv_feat_arr = relv_feat_arr[relv_indices]

    # plot_tsne_cluster(n_clusters, labels, True, tar_name, concat_tar_features, prev_src_feat, curr_src_features, centroids)
    # plot_tsne_cluster(n_clusters, labels, False, tar_name, concat_tar_features, relv_feat_arr, curr_src_features, centroids)
    

## ----------------------------------- -----------------------------------------##
# Separate the transformed data back into original features, second features, and centroids
    # original_features_2d = combined_2d[:concat_tar_features.shape[0]]
    # second_features_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + concat_src_features.shape[0]]
    # centroids_2d = combined_2d[concat_tar_features.shape[0] + concat_src_features.shape[0]:]

    # # Plot the t-SNE visualization
    # plt.figure(figsize=(12, 8))

    # # Plot the original features with circles
    # plt.scatter(original_features_2d[:, 0], original_features_2d[:, 1], c=labels[:concat_tar_features.shape[0]], marker='o', alpha=0.6, label='Original Features')

    # # Plot the second features with squares
    # plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c=labels[concat_tar_features.shape[0]:], marker='s', alpha=0.6, label='Second Features')

    # # Plot the centroids with a distinct marker
    # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')
# ------------------------------------ --------------------------------------##
    # features_2d = tsne.fit_transform(concat_tar_features.cpu().numpy())

    # Separate the transformed features and centroids
    # *** Giving equal size to every features BCZ of same data in all three feature map. Note: change this in case of selecting diff prev top-N
    # features_2d = combined_2d[:concat_tar_features.shape[0]]
    # features_2d = combined_2d[:combined_feat.shape[0]]
    # centroids_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + centroids.shape[0]]
    # second_features_2d = combined_2d[concat_tar_features.shape[0] + centroids.shape[0]:]

    # prev_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0]]
    # second_features_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0] + concat_tar_features.shape[0]]
    # centroids_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]+concat_tar_features.shape[0]:]
    

    # Plot the t-SNE visualization
    # plt.figure(figsize=(10, 8))
    # for i in range(n_clusters):
    #     # Select points belonging to the current cluster
    #     cluster_points = features_2d[labels == i]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Target Cluster {i}', alpha=0.6)

    # Plot the centroids
    # Plot the centroids in a distinct color and marker
    # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, label='Centroids')

    # plt.scatter(prev_2d[:, 0], prev_2d[:, 1], c='green', marker='o', alpha=0.3, label='Previous Relevent Features')

    # Plot the second set of features in a different color
    # plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c='gray', marker='o', alpha=0.3, label='Second Features')

    # plt.title('t-SNE Visualization of Clusters')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.legend()
    # tsne_folder = 'relv_samples_clusters/'+ tar_name.split('/')[0]
    # if not os.path.exists(tsne_folder):
    #     os.makedirs(tsne_folder)

    # file_name = tsne_folder+'/'+tar_name.split('/')[1]+'_BE.png' if is_before_adapt else tsne_folder+'/'+tar_name.split('/')[1]+'AF_.png'
    # plt.savefig(file_name)
    # plt.show()
    
    return relv_feat_arr, relv_src_data_arr, relv_src_label_arr

def create_relv_src_clus_cent(srcs_sub_loader, tar_sub_loader, model, prev_relv_samples_struc, relv_sample_count, tar_name, is_before_adapt=False):
    model.eval()

    concat_tar_features = []
    curr_src_features = []
    curr_src_data = []
    curr_src_label = []
    updated_relv_samples_struc = None

    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_label = sources
            src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
            src_feat = model.forward_features(src_data)
            curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 
            curr_src_data.extend(src_data.detach().cpu().numpy())
            curr_src_label.extend(src_label.detach().cpu().numpy())

            tar_data, _ = target
            tar_data = tar_data.cuda().float()
            tar_feat = model.forward_features(tar_data)
            concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

    # Define the number of clusters
    n_clusters = 2

    concat_tar_features = concat_tar_features.cpu().numpy()
    curr_src_features = curr_src_features.cpu().numpy()

    # Fit the model and get the cluster labels
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
    kmeans.fit(concat_tar_features)
    labels = kmeans.labels_ 
    centroids = kmeans.cluster_centers_

    # print("\nTarget Centroid:", centroids)
    kmeans_curr_src = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
    kmeans_curr_src.fit(curr_src_features)
    labels_curr_src = kmeans_curr_src.labels_
    centroids_curr_src = kmeans_curr_src.cluster_centers_

    # Step 3: Calculate distances with every sample in new subject with its own centroids
    B_distances = cdist(curr_src_features, centroids_curr_src, 'euclidean')
    B_min_distances = B_distances.min(axis=1) # it picks the smallest distance of every point to the centroids 
    # Include features, given labels (B_L), and K-means labels in B_results
    B_results = [(curr_src_data[i], curr_src_features[i], curr_src_label[i], B_min_distances[i]) for i in range(len(curr_src_data))]
    B_sorted = sorted(B_results, key=lambda x: x[3])  # Sort by distance (now at index 4)

    # calculate the distance with the target centroid 
    distances_from_A = cdist([result[1] for result in B_sorted[:len(B_sorted)]], centroids, 'euclidean')
    distances_from_A = distances_from_A.min(axis=1)
    new_relv_samples_struc = [(B_sorted[i][0], B_sorted[i][1], B_sorted[i][2], distances_from_A[i]) for i in range(len(B_sorted))]

    '''
        Add up based on the distances from target centroid Prev Samples + New Samples

        prev_relv_samples_struc: store prev closest samples
        new_relv_samples_struc: store new src samples
        updated_relv_samples_struc: store updated list after combining with new_relv_samples_struc
    '''
    if prev_relv_samples_struc:
        # only taking first 500 samples that are most closer to src centroid and check if its closer to the target subject or not
        updated_prev_struc = prev_relv_samples_struc + new_relv_samples_struc[:500]
        updated_relv_samples_struc = sorted(updated_prev_struc, key=lambda x: x[3])

        prev_src_feat =[point[1] for point in prev_relv_samples_struc if point[1] is not None]
    else:
        updated_relv_samples_struc = new_relv_samples_struc[:500]
        prev_src_feat =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

    if len(updated_relv_samples_struc) < relv_sample_count:
        relv_sample_count = len(updated_relv_samples_struc) 

    relv_feat_arr =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

    # plot_tsne_cluster(n_clusters, labels, True, tar_name, concat_tar_features, np.array(prev_src_feat), curr_src_features, centroids)
    # plot_tsne_cluster(n_clusters, labels, False, tar_name, concat_tar_features, np.array(relv_feat_arr[:relv_sample_count]), curr_src_features, centroids)
    
    return updated_relv_samples_struc[:relv_sample_count]

def create_relv_src_clus_dbscan(srcs_sub_loader, tar_sub_loader, model, prev_relv_samples_struc, relv_sample_count, tar_name, is_before_adapt=False):
    model.eval()

    concat_tar_features = []
    curr_src_features = []
    curr_src_data = []
    curr_src_label = []

    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_label = sources
            src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
            src_feat = model.forward_features(src_data)
            curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 

            curr_src_data.extend(src_data.detach().cpu().numpy())
            curr_src_label.extend(src_label.detach().cpu().numpy())

            tar_data, _ = target
            tar_data = tar_data.cuda().float()
            tar_feat = model.forward_features(tar_data)
            concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 
    
    concat_tar_features = concat_tar_features.cpu().numpy()
    # concat_src_features = concat_src_features.cpu().numpy() # combining all the relevant + new src sb
    curr_src_features = curr_src_features.cpu().numpy()
    
    tar_centroids_dbscan, tar_dbscan_clusters, tar_labels_dbscan = apply_dbscan(concat_tar_features)
    src_centroids_dbscan, src_dbscan_clusters, src_labels_dbscan = apply_dbscan(curr_src_features)
        
    # Step 3: Calculate distances with every sample in new subject with its own centroids
    B_distances = cdist(curr_src_features, src_centroids_dbscan, 'euclidean')
    B_min_distances = B_distances.min(axis=1) # it picks the smallest distance of every point to the centroids 
    # Include features, given labels (B_L), and K-means labels in B_results
    B_results = [(curr_src_data[i], curr_src_features[i], curr_src_label[i], B_min_distances[i]) for i in range(len(curr_src_data))]
    B_sorted = sorted(B_results, key=lambda x: x[3])  # Sort by distance (now at index 4)

    # calculate the distance with the target centroid 
    distances_from_A = cdist([result[1] for result in B_sorted[:len(B_sorted)]], tar_centroids_dbscan, 'euclidean')
    distances_from_A = distances_from_A.min(axis=1)
    new_relv_samples_struc = [(B_sorted[i][0], B_sorted[i][1], B_sorted[i][2], distances_from_A[i]) for i in range(len(B_sorted))]

    '''
        Add up based on the distances from target centroid Prev Samples + New Samples

        prev_relv_samples_struc: store prev closest samples
        new_relv_samples_struc: store new src samples
        updated_relv_samples_struc: store updated list after combining with new_relv_samples_struc
    '''
    if prev_relv_samples_struc:
        # only taking first 500 samples that are most closer to src centroid and check if its closer to the target subject or not
        updated_prev_struc = prev_relv_samples_struc + new_relv_samples_struc[:100] # biovid=500  , UNBC=100
        updated_relv_samples_struc = sorted(updated_prev_struc, key=lambda x: x[3])

        prev_src_feat =[point[1] for point in prev_relv_samples_struc if point[1] is not None]
    else:
        updated_relv_samples_struc = new_relv_samples_struc[:100]
        prev_src_feat =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

    if len(updated_relv_samples_struc) < relv_sample_count:
        relv_sample_count = len(updated_relv_samples_struc) 

    relv_feat_arr =[point[1] for point in updated_relv_samples_struc if point[1] is not None]
    print("** ** Updated Prev Dic using DBSCAN ** **")

    # plot_dbscan_tsne(tar_dbscan_clusters, src_dbscan_clusters, True, tar_name, concat_tar_features, np.array(prev_src_feat), curr_src_features, tar_centroids_dbscan)
    # plot_dbscan_tsne(tar_dbscan_clusters, src_dbscan_clusters, False, tar_name, concat_tar_features, np.array(relv_feat_arr[:relv_sample_count]), curr_src_features, tar_centroids_dbscan)
    
    return updated_relv_samples_struc[:relv_sample_count]

def apply_dbscan(input_features):
    dbscan_eps = 5.5
    dbscan_minsam = 4
    while True:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minsam)
        dbscan_clusters = dbscan.fit_predict(input_features)
        if max(dbscan_clusters) > -1:
            break
        else:
            dbscan_eps = dbscan_eps - 1.0
            dbscan_minsam = dbscan_minsam - 1
            print("EPS reduce by 1:: ", dbscan_eps)
            print("Min Sample reduce by 1: ", dbscan_minsam)

    labels_dbscan = set(dbscan_clusters)
    labels_dbscan.discard(-1)
    centroids_dbscan = []
    for label in labels_dbscan:
        cluster_points = input_features[dbscan_clusters == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids_dbscan.append(centroid)
    
    return centroids_dbscan, dbscan_clusters, labels_dbscan

def make_clusters(data):
    clusters_by_label = {}
    # Iterate over each label and cluster the corresponding feature vectors
    for label in range(2):
        label_data = data[data[:, -1] == label]
        label_features = label_data[:, :-1]
        if len(label_features) > 0:
            # kmeans = KMeans(n_clusters=1, random_state=0).fit(label_features)
            clusters_by_label[label] = label_features

    # Print the resulting clusters by label
    # print(clusters_by_label)
    return clusters_by_label

def calculate_centroid(clusters_by_label):
    cluster_centers = {}
    for label in range(7):
        if label in clusters_by_label:
            cluster_centers[label] = np.mean(clusters_by_label[label], axis=0)

    # print(cluster_centers)
    return cluster_centers


def making_clusters(data, labels, clusters_by_label):
    # convert into array from tensor
    arr_label = labels.cpu().detach().numpy()
    for index in range(len(arr_label)):
        clusters_by_label[arr_label[index]] = [data[index]]
        # feat_memory_bank[label] = data[data[:, -1] == label]
    
    print(clusters_by_label)


def k_means_clustering(data, center, batch_size, clusters, mod_cluster, clusters_weights):
    data = data.cpu().detach().numpy()
    # new_centroids = centroids
    # run the k-means algorithm for a fixed number of iterations
    # for _ in range(10):
    # compute the distances from each point to each centroid
    distances = np.zeros((batch_size, clusters))
    for i in range(batch_size):
        for j in range(clusters):
            distances[i, j] = np.linalg.norm(data[i, :] - center[j] * clusters_weights[j]) if mod_cluster else np.linalg.norm(data[i, :] - center[f'{j}'])
    
    # assign each point to the closest centroid
    new_labels = np.argmin(distances, axis=1)
    
    # compute the new centroids for each cluster
    for j in range(7):
        mask = new_labels == j
        if np.sum(mask) > 0:
            center[f'{j}'] = np.mean(data[mask, :], axis=0)

    # centroids = new_centroids if batch_count == n_batch - 1 else centroids
    return new_labels, center
        

def spherical_kmeans(X, K=16, max_iter=100, tol=1e-3):
    # Normalize feature vectors
    X = X.cpu().detach().numpy()
    X_normalized = normalize(X, axis=1)
    # Initialize centroids randomly
    centroids = X_normalized[np.random.choice(X.shape[0], K, replace=False)]
    prev_loss = None
    for i in range(max_iter):
        # Compute distances between each point and each centroid
        distances = pairwise_distances(X_normalized, centroids, metric='cosine')
        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)
        # Compute loss
        loss = np.sum(np.min(distances, axis=1))
        if prev_loss is not None and abs(loss - prev_loss) < tol:
            break
        prev_loss = loss
        # Update centroids
        for j in range(K):
            centroid_indices = np.where(labels == j)[0]
            if len(centroid_indices) > 0:
                centroids[j] = np.mean(X_normalized[centroid_indices], axis=0)
    return labels, centroids


def fer_clusters(features_center, centers, batch_size, clusters):
    distances = np.zeros((16, clusters))
    for i in range(16):
        for j in range(clusters):
            # distances[i, j] = np.linalg.norm(features_center[i, :] - centers[f'{j}'])
            distances[i, j] = np.mean(centers[f'{j}'], axis=0) - np.mean(features_center[i, :], axis=0) 
    
    # assign each point to the closest centroid
    new_labels = np.argmin(distances, axis=1)
    return new_labels
    

def k_means(data, minmax, index, threshold):
    
    kmeans = KMeans(n_clusters=4, n_init=20, max_iter=10, random_state=0)
    kmeans.fit_predict(data)
    labels = kmeans.labels_

    _unique = np.unique(labels, return_counts=True)
    # large_cluster_indx = 0 if _unique[1][0] > _unique[1][1] else 1
    large_cluster_indx = np.unravel_index(np.argmin(_unique[1]), _unique[1].shape) if minmax else np.unravel_index(np.argmax(_unique[1]), _unique[1].shape) 
    larger_cluster = []
    for i in range(len(labels)):
        if index != 7:
            if labels[i] == 1 or labels[i] == 2:
                larger_cluster.append(data[i])
        else:
            larger_cluster.append(data[i])  # for target, select all data points (clusters)
    
    larger_cluster = np.squeeze(larger_cluster)

    if index != 7:
        # calculate center and eliminate points whose distance from the center exceeds a threshold value
        center = np.mean(larger_cluster, axis=0)
        # Calculate the distances between every point and the center
        distances = np.linalg.norm(larger_cluster - center, axis=1)
        # Set a threshold value
        # threshold = 5.0 # for PCA value = 1.0
        mask = distances <= threshold
        # Eliminate the points whose distances from the center exceed the threshold
        filtered_arr = larger_cluster[mask]
    else:
         # calculate center and eliminate points whose distance from the center exceeds a threshold value
        center = np.mean(larger_cluster, axis=0)
        # Calculate the distances between every point and the center
        distances = np.linalg.norm(larger_cluster - center, axis=1)
        # Set a threshold value
        threshold = 8.0
        mask = distances <= threshold
        # Eliminate the points whose distances from the center exceed the threshold
        filtered_arr = larger_cluster[mask]
        # filtered_arr = larger_cluster

    return filtered_arr

def remove_outlier(larger_cluster, threshold):
    
    # calculate center and eliminate points whose distance from the center exceeds a threshold value
    center = np.mean(larger_cluster, axis=0)
    # Calculate the distances between every point and the center
    distances = np.linalg.norm(larger_cluster - center, axis=1)
    # Set a threshold value
    # threshold = 5.0 # for PCA value = 1.0
    mask = distances <= threshold
    # Eliminate the points whose distances from the center exceed the threshold
    filtered_arr = larger_cluster[mask]

    return filtered_arr

def cal_pca_source_mean_target(clusters):

    _src_mean = []
    target_points = []
    minmax = [True, True, True, False, True, False, True, False]
    for i in range(len(clusters)):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(clusters[i])

        pca_data = k_means(pca_data, minmax[i], i, threshold=1.0)

        if i != 7:
            cluster_mean = np.mean(pca_data, axis=0)
            _src_mean = np.column_stack(cluster_mean) if i == 0 else np.concatenate((_src_mean, np.column_stack(cluster_mean)), axis=0)
        else:
            target_points = np.row_stack(pca_data)


        # Generate blobs in the reduced 2D space
        # blobs, _ = make_blobs(n_samples=math.ceil(len(pca_data)/2), centers=1, random_state=0)

    return _src_mean, target_points

def making_target_clusters(data, labels):
    
    cluster_dict = {}
    for i in range(7):
        clusters = []
        inside = False
        for j in range(len(data)):
            if labels[j] == i:
                clusters = np.column_stack(data[j]) if not inside else np.concatenate((clusters, np.column_stack(data[j])), axis=0)
                inside = True
        cluster_dict[i] = clusters
    
    return cluster_dict
