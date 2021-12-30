library(ggplot2)
library(cowplot)

url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data <- read.csv(url, header = FALSE)

head(data)
# Column names are missing

# Add column names
colnames(data) <- c(
  "age",
  "sex",# 0 = female, 1 = male
  "cp", # chest pain
  # 1 = typical angina,
  # 2 = atypical angina,
  # 3 = non-anginal pain,
  # 4 = asymptomatic
  "trestbps", # resting blood pressure (in mm Hg)
  "chol", # serum cholestoral in mg/dl
  "fbs",  # fasting blood sugar if less than 120 mg/dl, 1 = TRUE, 0 = FALSE
  "restecg", # resting electrocardiographic results
  # 1 = normal
  # 2 = having ST-T wave abnormality
  # 3 = showing probable or definite left ventricular hypertrophy
  "thalach", # maximum heart rate achieved
  "exang",   # exercise induced angina, 1 = yes, 0 = no
  "oldpeak", # ST depression induced by exercise relative to rest
  "slope", # the slope of the peak exercise ST segment
  # 1 = upsloping
  # 2 = flat
  # 3 = downsloping
  "ca", # number of major vessels (0-3) colored by fluoroscopy
  "thal", # this is short of thalium heart scan
  # 3 = normal (no cold spots)
  # 6 = fixed defect (cold spots during rest and exercise)
  # 7 = reversible defect (when cold spots only appear during exercise)
  "hd" # (the predicted attribute) - diagnosis of heart disease
  # 0 if less than or equal to 50% diameter narrowing
  # 1 if greater than 50% diameter narrowing
)

head(data)
str(data)
# "?" in data

# Convert "?" to NA
data[data == "?"] <- NA

# Factorise variable sex
data[data$sex == 0, ]$sex <- "F"
data[data$sex == 1, ]$sex <- "M"
data$sex <- as.factor(data$sex)
str(data)

# Factorise other variables
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
str(data)

# Convetrt variables ca and thal to integer and factorise
data$ca <- as.integer(data$ca)
data$ca <- as.factor(data$ca)

data$thal <- as.integer(data$thal)
data$thal <- as.factor(data$thal)
str(data)

# Replace 0 and 1 with "Healthy" and "Unhealthy" and factorise
data$hd <- ifelse(data$hd == 0, "Healthy", "Unhealthy")
data$hd <- as.factor(data$hd)
str(data)

# Determin the number of rows with "NA"
nrow(data[is.na(data$ca) | is.na(data$thal),])
data[is.na(data$ca) | is.na(data$thal), ]
nrow(data)

# Remove samples with "NA"
data <- data[!(is.na(data$ca) | is.na(data$thal)), ]
nrow(data)

# Quality control (boolean and categorical variables)
xtabs(~ hd + sex, data = data)
xtabs(~ hd + cp, data = data)
xtabs(~ hd + fbs, data = data)
# Possible problem: only 4 people with level 1
xtabs(~ hd + restecg, data = data)
xtabs(~ hd + exang, data = data)
xtabs(~ hd + slope, data = data)
xtabs(~ hd + ca, data = data)
xtabs(~ hd + thal, data = data)

# Simple model: predicting heart disease using only gender
logistic <- glm(hd ~ sex, data = data, family = "binomial")
summary(logistic)

female.log.odds <- log(25/71)
female.log.odds

male.log.odds.ratio <- log((112/89)/(25/71))
male.log.odds.ratio

ll.null <- logistic$null.deviance/-2
ll.proposed <- logistic$deviance/-2

# McFadden's Pseudo R^2
(ll.null - ll.proposed)/ll.null

# p-value of McFadden's Pseudo R^2
1 - pchisq(2*(ll.proposed - ll.null), df=1)
1 - pchisq((logistic$null.deviance - logistic$deviance), df = 1)

# Prediction 
predicted.data <- data.frame(
  probability.of.hd = logistic$fitted.values,
  sex=data$sex)

ggplot(data = predicted.data, aes(x=sex, y=probability.of.hd)) +
  geom_point(aes(color=sex), size = 5) +
  xlab("Sex") +
  ylab("Predicted probability of getting heart disease")

xtabs(~ probability.of.hd + sex, data = predicted.data)

# Complex model: predicting heart disease using all variables
logistic <- glm(hd ~ ., data = data, family = "binomial")
summary(logistic)

ll.null <- logistic$null.deviance/-2
ll.proposed <- logistic$deviance/-2

# McFadden's Pseudo R^2
(ll.null - ll.proposed)/ll.null

# p-value of McFadden's Pseudo R^2
1 - pchisq(2*(ll.proposed - ll.null), df = (length(logistic$coefficients) - 1))

predicted.data <- data.frame(probability.of.hd = logistic$fitted.values,
                             hd = data$hd)
predicted.data <- predicted.data[order(predicted.data$probability.of.hd,
                                       decreasing = FALSE), ]
predicted.data$rank <- 1:nrow(predicted.data)

ggplot(data = predicted.data, aes(x=rank, y=probability.of.hd)) +
  geom_point(aes(color=hd), alpha = 1, shape = 4, stroke = 2) +
  xlab("Index") +
  ylab("Predicted probability of getting heart disease")

ggsave("heart_disease_probabilities.pdf")