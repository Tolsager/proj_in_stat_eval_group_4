file <- "df3.csv"
data <- read.csv(file)
data <- data[-417, ]
data$person <- as.factor(data$person)
data$experiment <- as.factor(data$experiment)
response <- cbind(data$curve_length, log(data$sq_distance), data$avg_d_obj) #nolint

boxplot(data$curve_length ~ data$person, xlab = "Person", ylab = "Curve length", main = "Boxplot for curve length per person") #nolint

model <- lm(data$curve_length ~ data$person * data$experiment)
anova(model)
plot(model)
x <- resid(model)
hist(x, density = NULL, breaks = 20, col = "#da0d28", main = "Histogram with residuals and normal distribution", xlab = "Residuals")#nolint
normal_x <- seq(min(x), max(x), length.out = 100)
normal_density <- dnorm(normal_x, mean(x), sd(x))
par(new = TRUE)
plot(normal_x, normal_density, col = "blue",
    type = "l", xaxt = "n", yaxt = "n", xlab = "", ylab = "", lwd = 2.0)


plot(ecdf(x), verticals = T, main = "Cummulative desity function comparison", xlab = "Residuals") #nolint
xseq <- seq(0.9 * min(x), 1.1 * max(x), length.out = 100)
lines(xseq, pnorm(xseq, mean(x), sd(x)), col = 'red')
legend("bottomright", legend = c('Residuals', 'Normal dist.'), col = c("red", "black"), lty=1:1, cex = 2, #nolint
        text.width = 13.5)     #nolint


plot(x, ylab = "Residuals", main = "Residuals")

anova(lm(curve_length ~ person + experiment + experiment:person, data = data))

mean_coor <- rep(NA, 100)
mean_exp <- rep(NA, 16)
mean_persons <- matrix(NA, 16, 10)
for (p in 1:10) {
    for (e in 1:16) {
        for (i in 1:100) {
            mean_coor[i] <- mean(unlist(c(data[data$person == p & data$experiment == e, 108:207])[i])) #nolint
        }
        mean_exp[e] <- mean(mean_coor)
    }
    mean_persons[, p] <- mean_exp
}
mean_persons
boxplot(mean_persons, main = "Boxplot of mean y value for each person", ylab = "Mean y value", xlab = "Person") #nolint


#A million boxplots
library(stringi)
boxplot(data$curve_length ~ data$person * data$experiment)


string <- "Boxplot 1.jpeg"
stri_sub(string, 9, 9) <- 7
string

for (i in 1:16) {
    string <- "Boxplot of curve length per person for experiment "
    string2 <- "Boxplot 1.jpeg"
    stri_sub(string, 51, 50) <- i
    stri_sub(string2, 9, 9) <- i
    jpeg(string2, width = 450, height = 350)
    boxplot(data$curve_length[data$experiment == i] ~ data$person[data$experiment == i],  xlab = "Person", ylab = "Curve length", main = string) #nolint
    dev.off()
    }

