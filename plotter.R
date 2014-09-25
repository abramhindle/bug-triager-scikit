library(zoo)

v<-read.csv("out/document_topic_map.csv", header=T)
vd<-read.csv("out/doc_date.csv", header=T)
length(v$Doc[v$Doc == vd$Doc]) == length(v$Doc)
v$utime = vd$utime
v$date = vd$date
v$day = vd$day

colors <- c(
            "#ff0000",
            "#ffbfbf",
            "#ff8000",
            "#ffff00",
            "#00ff00",
            "#00ffff",
            "#0000ff",
            "#8000ff",
            "#bf0080",
            "#40ffff",
            "#ffffff",
            "#bfbfbf",
            "#ffffbf",
            "#8000ff",
            "#800000",
            "#008080",
            "#bfffbf",
            "#bf4000",
            "#4080ff",
            "#bf8000"
            )



plot(rollmean(v$T1,100))

dateRange <- function(days) {
  days <- as.Date(days)
  as.Date(min(days):max(days))
}

everynth <- function(l , n) {
  l[1:length(l) %% n == 0]
}

avgperday <- function(v,vday,column, minreplace=0) {
  days <- unique(vday)
  x <- sapply(days, function(day) { mean(v[vday==day,column]) })
  x[!is.finite(x)] <- minreplace
  z <- c()
  z$days <- days
  z$x <- x
  z
}
polyplot <- function(l,xlabels=c(),
                     ourTitle = "title",
                     color = "aquamarine",
                     ylabel = "Avg. Topic Weight",
                     xlabel = "Time",
                     borderColor = "black",
                     nlabels = 20) {
  below <- 1
  lx <- c(1:(length(l)))
  xx <- c(lx,rev(lx))
  ml <- min(l)
  maxl <- max(l)
  yy <- c(l, l*0+ml)
  plot(c(),xlim=c(1,length(l)),ylim=c(ml,maxl),xaxt="n",ylab=ylabel,xlab=xlabel,main=ourTitle)
  if (length(l) > nlabels) {
    n <- length(l)
    nlx <- round(lx*(n/nlabels))
    axis(below, nlx, xlabels[nlx])
  } else {
    axis(below, lx, xlabels)
  }
  #polygon(xx,yy,col="aquamarine",border="darkgreen",lty=2)
  polygon(xx,yy,col=color,border=borderColor,lty=1)  
}

zeromean <- function(l) {
    if (length(l)==0) {
        return(0)
    } else {
        return(mean(l))
    }
}

topics <- names(v)[2:21]


svg("Everything.svg",width=17,height=22)
par(mfrow=c(10,2))
par(oma=c(1,1,1,1))
par(mar=c(2,5,1,1))
par(cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
sapply(c(1:20), function(topic) {
    ttopic = paste(c("T",topic),collapse="",sep="")
    file <- paste(c(ttopic, ".svg"),collapse="",sep="")
    #svg(file,width=8.5,height=4)
    n <- 14
    ldays <- everynth(dateRange(v$day),n)
    v$dd <- as.Date(v$day)
    f <- zeromean
    vals <- sapply(ldays, function(day) { f(v[v$dd >= as.Date(day) & v$dd < as.Date(day)+n, ttopic]) })
    polyplot(vals, ldays, ourTitle = ttopic, color=colors[topic])
    lines(lowess(vals))
    #dev.off()
})
dev.off()
