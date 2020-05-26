
# first dataset-------

ts.length <- 100
lags <- 0:4
nr.series <- length(lags)
rf <- g(1:(ts.length+max(lags)))


df <- ts.generator(ts.length, rf, lags, f1)
df <- cbind(df,
            ts.generator(ts.length, rf, lags, f2))
df <- cbind(df,
            ts.generator(ts.length, rf, lags, f3))

df <- cbind(df,
            ts.generator(ts.length, rf, lags, f4))

nr.classes <- 4

names(df) <- as.character(1:(nr.classes * nr.series))
true_cluster <- c(rep(1, nr.series), rep(2,nr.series), rep(1, nr.series), rep(2, nr.series))

colSums(df<0)
# i <- i + 1

write.csv(df, file="Thesis_prep/Data/synthetic_data.csv", row.names = F)

res.df <- get.residuals(df)

write.csv(res.df, file="Thesis_prep/Data/synthetic_data_res.csv", row.names = F)


plot_mult(data=df[, 1:nr.series])
plot_mult(data=df[, (nr.series+1):(2*nr.series)])
plot_mult(data=df[, (2*nr.series+1):(3*nr.series)])

plot_mult(df[,1], df[,6], df[,11], df[,16])
plot_mult(df[,1], df[,2], df[,3], df[,4], df[,5])


a <- ccf(df[,1], df[,2])
a

# second dataset ------
ts.length <- 100

df <- c1(100, seed=30)
df <- cbind(df, c2(100, seed=40))
df <- cbind(df, c1(100, seed=45))
df <- cbind(df, c2(100, seed=50))

nr.classes <- 4

names(df) <- as.character(1:(nr.classes * nr.series))

colSums(df<0)



plot_mult(df[,1], df[,6], df[,11], df[,16])
plot_mult(df[,1], df[,2], df[,3], df[,4], df[,5])

write.csv(df, file="Thesis_prep/Data/synthetic_data2.csv", row.names = F)

res.df <- get.residuals(df)

write.csv(res.df, file="Thesis_prep/Data/synthetic_data_res2.csv", row.names = F)

df[,1]

true_cluster <- c(rep(1, nr.series), rep(2,nr.series), rep(3, nr.series), rep(4, nr.series))

library('TSclust')

cluster_eval(df, dist.method='AR.PIC', linkage.method = 'single', true_cluster)
cluster_eval(df, dist.method='AR.MAH', linkage.method = 'single', true_cluster)
cluster_eval(df, dist.method='AR.MAHP', linkage.method = 'single', true_cluster)


log_df <- as.data.frame(diff(log(as.matrix(df))))
cluster_eval(log_df, dist.method='AR.PIC', linkage.method = 'single', true_cluster)
cluster_eval(log_df, dist.method='AR.MAH', linkage.method = 'single', true_cluster)

# cluster plotting --------
library(ggplot2)

theme_set(theme_classic())

df <- read.csv('Thesis_Prep/Results/clusters_complete_3.csv')

df$clusters <- df$clusters + 1
df$gt_num <- as.character(as.numeric(df$gt))
# Histogram on a Continuous (Numeric) Variable
plot.hist(df, save.fig=T, fig.size=c(6,4))

table(df$clusters)

# change binwidth
  # labs(title="Histogram of Clusters", 
  #      subtitle="Engine Displacement across Vehicle Classes")  

    # geom_histogram(aes(fill=gt_num), 
    #                bins=10, 
    #                col="black", 
    #                size=.1) +   # change number of bins
  # labs(title="Histogram with Fixed Bins", 
  #      subtitle="Engine Displacement across Vehicle Classes") 



# graph plotting -----
library(ape)
library(ggnetwork)
library(ggplot2)
library(igraph)

# diss_data <- read.csv('Thesis_Prep/diss_mat.csv', header=F)
# sectors <- read.csv('Thesis_Prep/Data/sectors.csv', header=F)
# revenues <- read.csv('Thesis_Prep/Data/revenues.csv', header=F)
# rev_types <- read.csv('Thesis_Prep/Data/rev_types.csv', header=F)

diss_data <- read.csv('Thesis_Prep/diss_mat2.csv', header=F)
sectors <- read.csv('Thesis_Prep/Data/sectors2.csv', header=F)
revenues <- read.csv('Thesis_Prep/Data/revenues2.csv', header=F)
rev_types <- read.csv('Thesis_Prep/Data/rev_types2.csv', header=F)

# diss_data2 <- read.csv('Thesis_Prep/diss_mat3.csv', header=F)


# as.numeric(table(sectors))
sectors <- sectors$V1
revenues <- revenues$V1
rev_types <- rev_types$V1

# names(diss_data) <- sectors

gr <- graph.adjacency(as.matrix(diss_data),
                      mode='undirected',
                      weighted = T)
mstree <- igraph::mst(gr)


# V(mstree)$shape <- as.character(rev_types)
hist(E(mstree)$weight)
hist(E(mstree2)$weight)

E(mstree)$weight == E(mstree2)$weight

secs <- as.numeric(sectors)
secs <- factor(secs, levels=as.character(seq(1, 10, 1)))

final_mstree <- plot.mst(mstree, names=secs,
                         pallete='Paired',
                         shapes=NA,
                         sizes=NA,
                         threshold=0.8,
                         save.fig = T, 
                         fig.size=c(6,4))


length(E(final_mstree))

hist(E(final_mstree)$weight)

# permutation test -----
edges <- get.edgelist(final_mstree)
s0 <- get.pure.edges(edges)
s <- permutations(final_mstree, n=1000)

permutation.test(final_mstree, 10^3)

length(E(final_mstree))

hist(s)
max(s)
factorial(4)
mean(s)
sd(s)
zap
# names(diss_data) <- sectors
# 
# gr <- graph.adjacency(as.matrix(diss_data), mode='undirected', weighted = T)
# mstree <- igraph::mst(gr)
V(mstree) <- sectors[sample(length(sectors))]

V(mstree)$name <- as.character(sectors[sample(length(sectors))])


mstree <- relabel.vertices(mstree)

f <- plot.mst(a, threshold=NA)
f



levels(sectors$V1) <- 1:10

names(diss_data) <- sectors$V1[sample(length(sectors$V1))]

dist_mat <- as.dist(diss_data)
mstree <- ape::mst(dist_mat)
gr <- graph.adjacency(mstree, mode='undirected')

a <- get.edgelist(gr)
























ccf(res.df[,1], res.df[,2])

m1 <- auto.arima(df[,1], max.d = 1, max.q=0, seasonal = F)
tsdiag(m1)
res0 <- residuals(m1)

m2 <- auto.arima(df[,2], max.d = 1, max.q=0, seasonal = F)
tsdiag(m2)
res1 <- residuals(m2)


m3 <- auto.arima(df[,6], max.d = 1, max.q=0, seasonal = F)
tsdiag(m3)
res2 <- residuals(m3)

m4 <- auto.arima(df[,7], max.d = 1, max.q=0, seasonal = F)
tsdiag(m4)
res3 <- residuals(m4)

m5 <- auto.arima(df[,11], max.d = 1, max.q=0, seasonal = F)
tsdiag(m5)
res4 <- residuals(m5)

m6 <- auto.arima(df[,12], max.d = 1, max.q=0, seasonal = F)
tsdiag(m6)
res5 <- residuals(m6)


ccf(res0, res1)
ccf(res0, res2)
ccf(res0, res3)
ccf(res0, res4)
ccf(res0, res5)



ts1 <- df[,1]
plot(ts1, type='l')
plot(diff(ts1), type='l')
acf(diff(ts1))
pacf(diff(ts1))
m1 <- arima(ts1, order=c(8,1,0))

step()

# plot(c(df[,1], predict(m1, 1)$pred))

tsdiag(m1, gof.lag = 30)
res1 <- residuals(m1)
res2 <- residuals(m1)
res3 <- residuals(m1)

ccf(res1, res2)
ccf(res1, res3)
ccf(res2, res3)
ccf(res2, res4)
a <- ccf(df[,1], df[,2])

step(object = "arima")




ts2 <- df[,14]
plot(ts2, type='l')
acf(ts2)
pacf(ts2)
m2 <- arima(ts2, order=c(8,0,0))
tsdiag(m2, gof.lag = 30)
res3 <- residuals(m2)

ts3 <- df[,6]
plot(ts3, type='l')
acf(diff(ts3))
pacf(diff(ts3))
m3 <- arima(ts3, order=c(8,1,0))
tsdiag(m3, gof.lag = 30)
res4 <- residuals(m3)

ccf(res0, res4)



# plot(diff(ts2), type='l')
# acf(diff(ts2))
# pacf(diff(ts2))
# m2 <- arima(ts2, order=c(5,1,0))

tsdiag(m2, gof.lag = 10)
res2 <- residuals(m2)

cor(res1, res2)
cor(ts1, ts2)

m2$coef[1] * ts2[100] + m2$coef[2] * (1-m2$coef[1])


predict(m2)
# ts1 <- f1(6:(ts.length+5), rf)
# ts2 <- f2(1:ts.length, rf)
# ts3 <- f3(1:ts.length, rf)










# lags <- 1:5#seq(5,30,5)
lags <- seq(1,9,2)
nr.series <- length(lags) #diff(range(lags))+1

# ts.sim <- sim.pos(arima.sim(list(order = c(1,1,0), ar = 0.85),
#                             n = 99) + rnorm(100, sd=3))

set.seed(100)

# dat <- read.csv('Thesis_Prep/AAPL.csv')
# 
# ts.sim <- dat$Close[1000:1200]




plot(ts.sim,type='l')
# plot(diff(log(ts.sim)))


df <- ts.generator(ts.sim, lags=lags, class.fun=class1)
df <- cbind(df,
            ts.generator(ts.sim, lags=lags, class.fun=class1.2))
df <- cbind(df,
            ts.generator(ts.sim, lags=lags, class.fun=class2))

names(df) <- as.character(1:(3 * nr.series))


# write.csv(df, file="Thesis_prep/Data/synthetic_data.csv", row.names = F)


plot_mult(data=df[, 1:5])
plot_mult(data=df[,c(1, 6:10)])
plot_mult(data=df[,c(1, 11:15)])



# 
# boxplot(cor(df)[lower.tri(cor(df))])
# hist(cor(df)[lower.tri(cor(df))])

# plot(ts.sim)
# plot(exp(cumsum(diff(log(ts.sim)))),type='l')
# plot(cumsum(diff(ts.sim)+rnorm(99)),type='l')

# data <- read.csv('Thesis_Prep/Data/stock_prices.csv')
# 
# keep.cols <- colSums(!is.na(data)) > nrow(data)*0.98
# data1 <- data[,unname(keep.cols)]
# data1 <- data1[, grepl('_Close', names(data1))]
# ts.sim <- na.omit(data1[, 5])


# plot_mult(data=df[, 1:6])
# plot_mult(data=df[,c(1, 7:12)])
# plot_mult(data=df[,c(1, 13:18)])




# true_cluster <- c(rep(1,2 * nr.series), rep(2,nr.series))

# library('TSclust')


cluster_eval(df, dist.method='CID', linkage.method = 'average', true_cluster)

cluster_eval(df, dist.method='AR.MAH', linkage.method = 'average', true_cluster)

diss.AR.MAH(df[,1],df[,2])


measures <- c('EUCL', 
              'DTW', 'CORT', 'COR',
              'ACF', 'PACF', 'PER',
              'INT.PER', 'SPEC.LLR',
              'DWT', 'AR.PIC', 'AR.MAH',
              'AR.LPC.CEPS', 'CID',
              'PDC', 'CDM', 'NCD')
# out <- vector()
# 
# for (m in measures){
#   out <- append(out, cluster_eval(df,
#                                   method = m,
#                                   true_cluster))
# }
# 
# out

plot(h.clust)

diss.COR(df[,1], df[,2])

sqrt(2*(1-cor(df[,1], df[, 2])))

cross.cor <- function(ts1, ts2, max.lag){
  n <- length(ts1)
  out <- vector()
  for (i in 1:max.lag){
    out <- append(out, cor(ts1[1:(n-i+1)], ts2[i:n]))
    out <- append(out, cor(ts2[1:(n-i+1)], ts1[i:n]))
  }
  # print(out)
  return(max(out))
}

cor(df[,2], df[,3])

cross.cor(df[,3], df[,2], 20)
cross.cor(diff(log(df[,1])), diff(log(df[,5])), 20)
ccf.vals <- ccf(df[,1], df[,7])
ccf.vals
# data <- read.csv('Thesis_Prep/Data/covid_data.csv')
# head(data)
# 
# df <- reshape(data[,1:3],
#               idvar="date",
#               timevar="location",
#               direction="wide")
# 
# a <- strsplit(names(df), split = 'new_cases.')
# for (i in 2:ncol(df)){
#   names(df)[i] <- a[[i]][2]
#   }

out <- diss(df, 'COR')

df[,1]
