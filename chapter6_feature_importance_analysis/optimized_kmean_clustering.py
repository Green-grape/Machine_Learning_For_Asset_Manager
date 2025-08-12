import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


class KMeansClustering:
    def __init__(
        self,
        distance_metric="euclidean",
    ):
        self.distance_metric = distance_metric
        if distance_metric == "euclidean":
            self.distance_func = lambda corr: np.sqrt(0.5 * (1 - corr.fillna(0)))
        else:
            raise NotImplementedError(
                f"Distance metric {distance_metric} is not implemented."
            )

    def __cluster_kmeans_base(self, corr: pd.DataFrame, max_num_clusters=10, n_init=10):
        """
        n_init을 여러번 하고 최대 max_num_clusters만큼 cluster를 만들어보면서 k-mean의 initialization 문제를 최소화하고 최적의 cluster 개수를 선정하여 결과를 반환하는 함수
        """
        x = self.distance_func(corr)

        best_cluster_quality = None
        best_sil = None
        best_kmeans = None
        for _ in range(n_init):  # multi initialization
            for i in range(2, max_num_clusters + 1):
                kmeans = KMeans(n_clusters=i, n_init=1).fit(x)
                sil = silhouette_samples(x, kmeans.labels_)
                cluster_quality = sil.mean() / sil.std()  # cluster score
                if (
                    best_cluster_quality is None
                    or best_cluster_quality < cluster_quality
                ):
                    best_cluster_quality = cluster_quality
                    best_sil = pd.Series(sil, index=x.index)
                    best_kmeans = kmeans

        sorted_idx = np.argsort(best_kmeans.labels_)
        ret_corr = corr.iloc[sorted_idx]  # reorder rows by cluster result
        ret_corr = corr.iloc[:, sorted_idx]  # reorder columns by cluster result

        cluster_dict = {
            i: corr.columns[np.where(best_kmeans.labels_ == i)[0].tolist()]
            for i in np.unique(best_kmeans.labels_)
        }
        return ret_corr, cluster_dict, best_sil

    def cluster_kmeans_top(self, corr: pd.DataFrame, max_num_clusters=None, n_init=10):
        """
        cluster_kmeans_base에 더하여 cluster마다 결과가 좋지 못했던 cluster만 모아서 재학습을 시켜 전반적인 cluster quality를 올리는 알고리즘
        """

        def override_cluster_result(
            corr: pd.DataFrame, cluster_dict_ori: dict, cluster_dict_new: dict
        ):
            new_cluster_dict_ret = {}
            for label in cluster_dict_ori.keys():
                new_cluster_dict_ret[len(new_cluster_dict_ret)] = list(
                    cluster_dict_ori[label]
                )
            for label in cluster_dict_new.keys():
                new_cluster_dict_ret[len(new_cluster_dict_ret)] = list(
                    cluster_dict_new[label]
                )  # override

            # reindex correlation matrix
            new_idx = [
                idx
                for i in new_cluster_dict_ret.keys()
                for idx in new_cluster_dict_ret[i]
            ]
            corr_new = corr.loc[new_idx, new_idx]

            # recalculate sil score
            x = self.distance_func(corr)  # distance of features
            kmeans_lables = np.zeros(len(x.columns))
            for i in new_cluster_dict_ret.keys():
                cur_idx = [x.index.get_loc(k) for k in new_cluster_dict_ret[i]]
                kmeans_lables[cur_idx] = i
            sil_new = pd.Series(
                silhouette_samples(x.values, kmeans_lables), index=x.index
            )
            return corr_new, new_cluster_dict_ret, sil_new

        if max_num_clusters is None:
            max_num_clusters = corr.shape[0] - 1

        ret_corr, cluster_dict, best_sil = self.__cluster_kmeans_base(
            corr, max_num_clusters, n_init
        )

        cluster_quality_dict = {
            label: (
                np.mean(best_sil[cluster_dict[label]])
                / np.std(best_sil[cluster_dict[label]])
                if np.std(best_sil[cluster_dict[label]]) != 0
                else 0
            )
            for label in cluster_dict.keys()
        }

        mean_cluter_quality = np.sum(list(cluster_quality_dict.values())) / len(
            cluster_quality_dict
        )

        bad_cluster_labels = [
            label
            for label in cluster_quality_dict.keys()
            if cluster_quality_dict[label] < mean_cluter_quality
        ]

        if len(bad_cluster_labels) <= 1:
            return ret_corr, cluster_dict, best_sil
        else:
            bad_cluster_data_index = [
                data_index
                for label in bad_cluster_labels
                for data_index in cluster_dict[label]
            ]
            bad_data_corr = corr.loc[bad_cluster_data_index, bad_cluster_data_index]
            mean_bad_cluster_quality = np.mean(
                [cluster_quality_dict[label] for label in bad_cluster_labels]
            )

            recluster_corr, recluster_cluster_dict, recluster_sil = (
                self.cluster_kmeans_top(
                    bad_data_corr,
                    min(max_num_clusters, bad_data_corr.shape[0] - 1),
                    n_init,
                )
            )

            # remake new output
            corr_new, cluster_dict_new, silh_new = override_cluster_result(
                corr,
                {
                    label: cluster_dict[label]
                    for label in cluster_dict.keys()
                    if label not in bad_cluster_labels
                },
                recluster_cluster_dict,
            )

            mean_new_cluster_quality = np.mean(
                [
                    (
                        np.mean(silh_new[cluster_dict_new[label]]) / std
                        if (std := np.std(silh_new[cluster_dict_new[label]])) != 0
                        else 0
                    )
                    for label in cluster_dict_new.keys()
                ]
            )

            if mean_new_cluster_quality <= mean_bad_cluster_quality:  # cluster 개선 X
                return ret_corr, cluster_dict, best_sil
            else:  # cluster 개선 O
                return corr_new, cluster_dict_new, silh_new
