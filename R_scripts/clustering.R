library(TSclust)
library(forecast)

cluster_eval <- function(df, dist.method, linkage.method, true_cluster, plot=T){
  if (dist.method == 'AR.MAH'){
    diss_mat <- diss(df, dist.method)$statistic
  } else if(dist.method == 'AR.MAHP'){
    diss_mat <- diss(df, 'AR.MAH')$p_value
  } else {
    diss_mat <- diss(df, dist.method) 
  }
  dendrogram <- hclust(diss_mat, method=linkage.method)
  if (plot){
    plot(dendrogram)
  }
  clustering <- cutree(dendrogram, k=length(unique(true_cluster)))
  return(cluster.evaluation(true_cluster, clustering))
}


get.residuals <- function(df){
  out.df <- df
  for (i in 1:ncol(df)){
    model <- auto.arima(df[,i], max.d = 1,
                        max.q=0, seasonal = F)
    out.df[,i] <- residuals(model)
  }
  return(out.df)
}  
