library(classInt)
library(data.table)
library(dplyr)
library(lfe)
library(multcomp)
library(plotrix)
library(RColorBrewer)
library(readr)
library(reshape2)
library(sampling)
library(sp)
library(splines)

data <- read.csv('~/Downloads/monthly_GAM_heatwave_days_state_1960_2019_final.csv')
est_us <- read.csv("~/Downloads/estimate_hw_us_updated_final.csv")
est_us <- est_us[, -which(names(est_us) == 'X')]

data$yr <- as.numeric(as.character(data$year))
yrs <- 2000:2019
ind <- data$yr%in%yrs
data <- data[ind,]
bus <- weighted.mean(data$suicide_rate,data$pop)

Tus <- read.csv('~/Downloads/heatwave_days_delta.csv') %>% as_data_frame()

#Tus$deltaT <- Tus$deltaDays

quantile(Tus$deltaDays,probs=c(0.025,0.5,0.975))  #statistics reported in paper
#quantile(Tmex$deltaT,probs=c(0.025,0.5,0.975))  #statistics reported in paper

dUS <- as.matrix(est_us)%*%matrix(Tus$deltaDays,nrow=1)/bus*100

bxu <- c(dUS)
qu <- quantile(bxu,probs=c(0.025,0.5,0.975))
bxu[bxu<qu[1]] <- qu[1]
bxu[bxu>qu[3]] <- qu[3]

print(qu)

pp <- read_csv('~/Downloads/NCC2018-master/inputs/projections/PopulationProjections.csv')  #projections, medium variant
popc <- list()  #list to fill

## 2020 to 2100
p1 <- pp[pp[,3]=="United States of America",paste0(2020:2100)]
popc[[1]] <- as.numeric(gsub(" ","",unlist(p1)))*1000


yr=2020:2100

# excess suicides in US
dUS <- as.matrix(est_us)%*%matrix(Tus$deltaDays,nrow=1)*12  #additional deaths per 100,000 per year by 2100 in US

# assume this rate increases linearly between 2000 and 2100 due to linear temperature increase
# looping over years, calculating effect for each bootstrap x model
ex <- array(dim=c(dim(dUS),length(yr)))
for (y in 1:length(yr)) {
  z <- (yr[y]-2020)/(2100-2020)
  ex[,,y] <- dUS*z*popc[[1]][y]/100000
}
exs <- apply(ex,c(1,2),sum)  #total cumulative deaths
exc_US <- exs

# define colors
alpha = 0.03
cll <- apply(sapply("orange", col2rgb)/255, 2, 
             function(x) rgb(x[1], x[2], x[3], alpha=alpha)) 

# define 2050 distributions for boxplotting
bxu <- c(exs)
qu <- quantile(bxu,probs=c(0.025,0.5,0.975))
bxu[bxu<qu[1]] <- qu[1]
bxu[bxu>qu[3]] <- qu[3]
print(qu)

#define median for each year
med <- apply(ex,3,median)
ub <- 100

#Panel (b)
plot(yr,cumsum(ex[1,1,])/1000,type="l",ylim=c(0,ub),col=cll,las=1,ylab="excess suicides (thousands)",xlab="year",lwd=0.3,xlim=c(2020,2102),cex.axis=1.5,cex.lab=1.5,axes=F)
for (j in 1:dim(ex)[1]) {
  for (i in 1:dim(ex)[2]) {
    lines(yr,cumsum(ex[j,i,])/1000,col=cll,lwd=0.3)
  }}

lines(yr,cumsum(med)/1000,col="black",lwd=2)
abline(h=0,lty=2)
boxplot(bxu/1000,horizontal=F,range=0,at=2102,add=T,col="orange",axes=F,boxwex=3,lty=1)

mtext("Excess suicides by 2100 (Heatwave), US", side=3, adj=0, line=0.5, cex=1.5, font=2)  #add title, left justified
axis(1,at=seq(2020,2100,10),labels=seq(2020,2100,10),cex.axis=1.5,las=1)
axis(2,at=seq(0,ub,20),labels=seq(0,ub,20),cex.axis=1.5,las=1)


