library(betareg)
library(emmeans)

BD <- read.csv("evenness/BD.tsv", sep="\t", header = TRUE, row.names = 1)
MDD <- read.csv("evenness/MDD.tsv", sep="\t", header = TRUE, row.names = 1)
SCZ <- read.csv("evenness/SCZ.tsv", sep="\t", header = TRUE, row.names = 1)
meta <- read.csv("metadata.tsv", sep="\t", header = TRUE, row.names = 1)

BDdata <- merge(BD, meta, by='row.names')
MDDdata <- merge(MDD, meta, by='row.names')
SCZdata <- merge(SCZ, meta, by='row.names')


model <- betareg(Y ~ Study + Disease | Disease, data = BDdata)
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <- betareg(Y ~ Study + Disease | Disease, data = MDDdata)
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <- betareg(Y ~ Study + Disease | Disease, data = SCZdata)
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")



