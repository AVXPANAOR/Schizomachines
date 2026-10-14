library('estimatr')
library(emmeans)


BD <- read.csv("faith/BD.tsv", sep="\t", header = TRUE, row.names = 1)
MDD <- read.csv("faith/MDD.tsv", sep="\t", header = TRUE, row.names = 1)
SCZ <- read.csv("faith/SCZ.tsv", sep="\t", header = TRUE, row.names = 1)
meta <- read.csv("metadata.tsv", sep="\t", header = TRUE, row.names = 1)

BDdata <- merge(BD, meta, by='row.names')[-c(126, 3, 42),]
MDDdata <- merge(MDD, meta, by='row.names')[-c(88, 499, 212, 449, 213, 490),]
SCZdata <- merge(SCZ, meta, by='row.names')[-c(199, 219, 337, 372, 83),]

model <-lm_robust(log(Y) ~ Study + Disease, data = BDdata, se_type ='HC3')
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <-lm_robust(log(Y) ~ Study + Disease, data = MDDdata, se_type ='HC3')
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <-lm_robust(log(Y) ~ Study + Disease, data = SCZdata, se_type ='HC3')
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

