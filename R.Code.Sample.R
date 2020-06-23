library(tidyverse)
library(mosaic)
library(ggplot2)
library(lubridate)
library(foreach)
library(LICORS)
library(randomForest)
library(rpart)
library(tree)
library(MLmetrics)
library(dplyr)
library(gamlr)
library(glmnet)
library(caret)
library(readr)
library(repr)
library(standardize)
library(plotmo)

pitching <- read.csv("~/Desktop/R/CleanPitching.csv")
pitching[is.na(pitching)] <- 0


pitch_k = pitching %>%
  group_by(k_9) %>%
  select(wpct, era, sv, ip, hr, bb_9, so, tbf, obp, sho, hb, gf) %>%
  summarize_all(mean)

#STRIKEOUT PCA
pca_k = prcomp(pitch_k, scale=TRUE)
plot(pca_k)
summary(pca_k)
round(pca_k$rotation[,1:8],2)

k_pc = merge(pitch_k, pca_k$x, by = "row.names")
k_pc = rename(k_pc, k9 = Row.names)
##pca graph
ggplot(k_pc) +
  geom_text(aes(x=PC2, y=PC5, label = k_9), size = 2)




#WHIP PCA
pitch_whip = pitching %>%
  group_by(whip) %>%
  select(wpct, sv, ip, hr, k_9, bb_9, so, tbf, obp, sho, hb, gf, era) %>%
  summarize_all(mean)

pca_whip = prcomp(pitch_whip, scale=TRUE)
plot(pca_whip)
summary(pca_whip)
round(pca_whip$rotation[,1:7],2)

whip_pc = merge(pitch_whip, pca_whip$x, by = "row.names")
whip_pc = rename(whip_pc, whip1 = Row.names)

# PC1 v PC2 is the best model, you can look at others but this one is good
ggplot(whip_pc) +
  geom_text(aes(x=PC1, y=PC2, label=whip), size = 2)


### Random Forests
n = nrow(pitching)
n_train = round(0.8*n)
n_test = n - n_train
train_cases = sample.int(n, n_train, replace=FALSE)
test_cases = setdiff(1:n, train_cases)
pitch_train = pitching[train_cases,]
pitch_test = pitching[test_cases,]

pitch_tree = rpart(k_9 ~ wpct + sv + ip + hr + whip + bb_9 + so +
                     tbf + obp + sho + hb + gf + era, method="anova", data=pitch_train,
                   control=rpart.control(minsplit=3, cp=1e-6, xval=20))
npitch = length(unique(pitch_tree$where))
# look at the cross-validated error
plot(pitch_tree)
yhat_test_tree1 = predict(pitch_tree, pitch_test)
mean((yhat_test_tree1 - pitch_test$k_9)^2) %>% sqrt


### RF graph
rf <- randomForest(k_9 ~ wpct + sv + ip + hr + whip + bb_9 + so +
                     tbf + obp + sho + hb + gf + era, data = pitch_train)


yhat_test_rf = predict(rf, pitch_test)
plot(predict(rf), pitch_train$k_9)
mean((yhat_test_rf - pitch_test$k_9)^2) %>% sqrt
RMSE.RF= mean((yhat_test_rf - pitch_test$k_9)^2) %>% sqrt

#Lasso

#### fix data
C1= read.csv("~/Desktop/R/CleanPitching.csv")
## rm ATL 
C1 <- C1[C1$team != "ATL", ]
### fit missiing with 0
C1[is.na(C1)] <- 0
C1=subset(C1, select = -c(X,Unnamed..0,player,name,position,year,team,k_9,bk))
C2=C1
##create a sparse matrix
c = sparse.model.matrix(so~ ., data=C1)[,-1]
y= na.omit(C1$so)
#### train sets
train=sample(1: nrow(c), nrow(c)/2)
test=(-train)
y.test=y[test]
grid = 10^ seq(10,-2, length = 100)
lasso= glmnet(c[train,],y[train],alpha=1,lambda = grid, standardize = TRUE )
cv.out = cv.glmnet(c[train,],y[train],alpha=1,standardize = TRUE )
lambda= cv.out$lambda.min
l.predict= predict(lasso,s=lambda, newx = c[test,])                   
RSME.Lasso= sqrt(mean((l.predict-y.test)^2))
out= glmnet(c[test,],y[test],alpha=1,lambda=grid)
lasso.co= predict(out,type = "coefficients",s=lambda)[1:38,]
lasso.co[lasso.co !=0]
line = (log(lambda))
line
lambda
## ols of selceted
fit= lm(so~wins+losses+era+g+gs+svo+h+r+er+whip+er+hr+bb+avg+cg+sho+hb+ibb+gf+hl+gidp+go+ao+wp+sb+cs+tbf+np+go_ao+obp+slg+h_9+ops+p_ip+wpct, data = C1)
fitp=predict(fit)
RSME.OLS= sqrt(mean((C1$so-fitp)^2))
#### wrong OLS fit
lin = lm(so~ ., data = C2[train,])
linp = predict(lin, data= C2[test,])
RSME.wrong= sqrt(mean((C1$so-linp)^2))
###graph
glmcoef<-coef(lasso,lambda)
coef.increase<-dimnames(glmcoef[glmcoef[,1]>0,0])[[1]]
coef.decrease<-dimnames(glmcoef[glmcoef[,1]<0,0])[[1]]
#get ordered list of variables as they appear at smallest lambda
allnames<-names(coef(lasso)[,
                            ncol(coef(lasso))][order(coef(lasso)[,
                                                                 ncol(coef(lasso))],decreasing=TRUE)])
#remove intercept
allnames<-setdiff(allnames,allnames[grep("Intercept",allnames)])
#assign colors
cols<-rep("gray",length(allnames))
cols[allnames %in% coef.increase]<-"green"      # higher mpg is good
cols[allnames %in% coef.decrease]<-"red"        # lower mpg is not

###rsme lambda graph

l.plot= plot(log(cv.out$lambda), cv.out$cvm , main = "MSE vs Lambda", ylab="MSE", xlab="Log(Lambda)",xlim=c(-7,7))
abline(v= line, col="purple")
text(xy.coords(-1.5, 2000),labels= c("Minimum Lambda"))
plot2 = plot_glmnet(lasso,label=5,s=lambda,col=cols, main = "Variable Significance Chart")

###



