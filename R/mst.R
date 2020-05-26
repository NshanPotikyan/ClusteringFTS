plot.hist <- function(df, save.fig=F, fig.size=c(6,4)){
  g <- ggplot(df, aes(clusters)) +
    scale_fill_brewer("Sector Id",
                      palette = "Paired",
                      breaks=as.character(seq(1, 10, 1))) 
  
  g <- g + geom_histogram(aes(fill=gt_num), 
                          binwidth = .5,
                          bins=10,
                          col="black", 
                          size=.1) +  
    scale_y_continuous(name="Frequency",
                       breaks=seq(0, 200, 25)) +
    scale_x_discrete(name="Clusters", 
                     limits=seq(1, 10, 1)) 
  print(g)
  if (save.fig){
    ggsave(filename = 'Thesis_Prep/pdf/hist.png',
           width=fig.size[1],height=fig.size[2])
  }
  
}

  
plot.mst <- function(mst, names=NA, pallete="Paired",
                     threshold=1, save.fig=F, fig.size=c(4,4)){
  if (is.na(threshold)){
    mstree <- mst
  } else {
    mstree <- delete.edges(mst, which(E(mstree)$weight > threshold))
  }
  
  if (!is.na(names[1])){
    V(mstree)$name <- names
  }
  
  co <- layout_with_fr(mstree, niter = 1000)
  gg <- ggnetwork(mstree, layout=co)

  g <- ggplot(gg, aes(x = x, y = y, xend = xend, yend = yend)) +
    geom_edges(color = "black", alpha = 0.4, curvature = 0.1) +
    nodes + scale_color_brewer("Sector Id",
                               palette = pallete,
                               breaks=as.character(seq(1, 10, 1))) +
    theme_blank() +
    theme(legend.background = element_blank(),
          legend.text = element_text(size = 8),
          legend.title = element_text(size = 8),
          legend.spacing.y = unit(0.1, 'cm'),
          legend.margin=margin(0, 0.2, 0, 0, "cm"),
          legend.position = 'right') 
  print(g)
  if (save.fig){
    ggsave(filename = 'Thesis_Prep/graph.png',
           width=fig.size[1],height=fig.size[2])
  }
  return(mstree)
}
  
  
get.pure.edges <- function(mat){
  return(sum(mat[,1] == mat[,2]))
}

  
relabel.vertices <- function(mst){
  vertices <- V(mst)$name
  V(mst)$name <- as.character(vertices[sample(length(vertices),
                                              replace = F)])  
  return(mst)
}


permutations <- function(mst, n=1000){
  out <- rep(0,n)
  for (i in 1:n){
    mstree <- relabel.vertices(mst)
    edges <- get.edgelist(mstree)
    out[i] <- get.pure.edges(edges)
  }
  write.csv(out,"Thesis_Prep/Results/null_distribution.csv", row.names = F)
  return(out)
}
  
permutation.test <- function(mst, nr.perm=1000){
  edges <- get.edgelist(mst)
  s0 <- get.pure.edges(edges)
  s <- permutations(mst, n=nr.perm)
  b <- sum(s >= s0)
  return((b+1) / (nr.perm+1))
}




