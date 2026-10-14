library(MASS)
library(emmeans)


BD <- read.csv("observed/BD.tsv", sep="\t", header = TRUE, row.names = 1)
MDD <- read.csv("observed/MDD.tsv", sep="\t", header = TRUE, row.names = 1)
SCZ <- read.csv("observed/SCZ.tsv", sep="\t", header = TRUE, row.names = 1)
meta <- read.csv("metadata.tsv", sep="\t", header = TRUE, row.names = 1)

BDdata <- merge(BD, meta, by='row.names')[-42,]
MDDdata <- merge(MDD, meta, by='row.names')[-c(486, 499, 490),]
SCZdata <- merge(SCZ, meta, by='row.names')[-c(219, 337),]


model <- glm.nb(Y ~ Study + Disease, data = BDdata, link = "log")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <- glm.nb(Y ~ Study + Disease, data = MDDdata, link = "log")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")

model <- glm.nb(Y ~ Study + Disease, data = SCZdata, link = "log")
summary(model)
emm <- emmeans(model, ~ Disease, type = "response")
pairs(emm, adjust = "none")



